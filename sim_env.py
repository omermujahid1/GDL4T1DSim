# ©Omer Mujahid
#importing libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

#importing and normalizing data
in_seq1 = pd.read_excel('insulin.xlsx')
in_seq2 = pd.read_excel('carbs.xlsx')
out_seq = pd.read_excel('bg.xlsx')
scaler = MinMaxScaler()
# scaler = StandardScaler()
# fit and transform in one step
normalized_insulin = scaler.fit_transform(in_seq1)
normalized_carbs = scaler.fit_transform(in_seq2)
normalized_bg = scaler.fit_transform(out_seq)

#closed-loop controller class
class PID_SMRC_IFB(object):
    """Base implementation of therapies (a.k.a. controllers)."""

    def __init__(self, name='Unknown', cgm_init=100, cr=100, cf=10, tdi=40, basalinsulin=1.0, bw=70):
        self.name = name
        self.announce_meals = True  # Announces meals?
        self.ts = 5

        # Patient specific information
        self.BW = bw  # Weight [kg]
        self.CR = cr  # Carbohydrates ratio [g/U]
        self.CF = cf  # Correction factor [mg/dl/U]
        self.TDI = tdi  # Total daily insulin [U]
        self.basalinsulin = basalinsulin  # Basal insulin [U/h]
        self.u_km1 = self.basalinsulin

        # Controller parameters
        self.Gref = 100  # [mg/dl]
        self.td = 90  # Derivative time [min]
        self.kp = (60 * self.TDI) / (self.td * 1500)  # Proportional gain [U/h]
        self._kdia = 0.03  # Time constant of the iob model [min^-1]
        self.IOBmax = (self.basalinsulin / 60 / self._kdia) * 2  # IOB upper limit [U]
        self.IOBmin = 0.0  # IOB lower limin [U]
        self.Lambda = 0.1  # Cutoff frequency of the SMRC filter
        self.tau = 10  # Sensitivity of the SMRC in front IOB changes [min]
        self.KIOB = 1  # IOB limit gain for day periods
        self.KIOB_night = 1  # IOB limit gain for night periods
        self.upper_w = 350  # [mg/dl]
        self.lower_w = 40  # [mg/dl]
        self.IOBmax_adaptation = True
        self.dt = 0.5
        self.internal_loop_step = int(self.ts / self.dt)

        # Controller lists for internal loop
        self.uk_hist = [0.0 for _ in range(self.internal_loop_step)]
        self.iob_hist = [0.0 for _ in range(self.internal_loop_step)]
        self.gref_hist = [0.0 for _ in range(self.internal_loop_step)]
        self.w_hist = [0.0 for _ in range(self.internal_loop_step)]
        self.ip_hist = [0.0 for _ in range(self.internal_loop_step)]

        # Controllar initialization parameteres
        self._glucose_km1 = cgm_init  # Initial CGM value
        self._IOBmax_km1 = self.IOBmax  # TODO This would eventually depend on the day/night period
        self._gref_filtered_km1 = self.Gref
        self._basal_km1 = self.basalinsulin

        # Plasma insulin model parameters
        self.ke = 0.138  # Insulin elimination from plasma (min^-1) [Hovorka et al 2004]
        self.V_I = 0.12  # Insulin distribution volume (L/kg) [Hovorka et al 2004]
        self.gamma = 0.42 * 60  # IFB gain parameter (L/h)

        # Attributes for IOB model
        self._init_model_iob()

        # Controller history
        self.history_ts = {'iob': [], 'insulin': [], 'iob_max': [], 'kiob': []}
        self.history_dt = {'uk_hist': [], 'iob_hist': [], 'gref_hist': [], 'w_hist': [], 'ip_hist': []}
        self.bolus = []
    def __repr__(self):
        return f"Controller PD+SAFE+IFB of patient {self.name}"

    def _init_model_iob(self):
        """Initial state for the internal controller iob model (equilibrium)"""
        C1eq = (self.basalinsulin / 60) / self._kdia
        C2eq = C1eq
        init_iob = C1eq + C2eq
        self._C1_km1 = init_iob / 2
        self._C2_km1 = init_iob / 2
        self._delta_C1_km1 = 0.0
        self._delta_C2_km1 = 0.0
        self._iob_km1 = self._C1_km1 + self._C2_km1 + self._delta_C1_km1 + self._delta_C2_km1
        self._deltaIpest_km1 = 0.0
        self.iob = self._iob_km1

    def run_one_step(self, cgm, meal_announcement=0.0, day=False):
        """Main run method of the controller."""

        # IOBmax limit adaptation for day/night periods
        if self.IOBmax_adaptation:
            if day:
                KIOB = self.KIOB
            else:
                KIOB = self.KIOB_night

            IOBmax = KIOB * self.IOBmax

        else:
            IOBmax = self.IOBmax

        dGdt = (self._glucose_km1 - cgm) / self.ts  # Glucose derivative [mg/dl/min]
        dIOBmax = (IOBmax - self._IOBmax_km1) / self.ts  # IOBmax derivative [U/min]

        if meal_announcement > 0.0:
            insulin_bolus = self._compute_bolus(cgm=cgm, cho_grams=meal_announcement)
        else:
            insulin_bolus = 0.0

        # Main control loop
        for i in range(self.internal_loop_step):
            self.estimate_iob(basal_insulin=self.basalinsulin, bolus_insulin=insulin_bolus,
                              controller_insulin=self.u_km1, k=i)  # update our internal step iob estimation
            deltaIpest_k = self._update_plasma_insulin()
            dIOBestk = (self.iob - self._iob_km1) / self.dt
         
            # Computation of sigma (SAFE)
            sigma_menos = ((IOBmax - (self.iob + self.tau * (dIOBestk - dIOBmax))) < 0)
            sigma_mas = ((self.IOBmin - (self.iob + self.tau * dIOBestk)) > 0)

            if sigma_menos:
                wk = self.upper_w
            elif sigma_mas:
                wk = self.lower_w
            else:
                wk = 0.0

            # Reference filtering
            Gref_filteredk = self._gref_filtered_km1 - self.dt * self.Lambda * (
                    self._gref_filtered_km1 - (self.Gref + wk))
            #print("REFERENCE GLUCOSE", Gref_filteredk)
            # PD control action [U/h]
            uk = self.kp * (cgm - Gref_filteredk) + self.kp * self.td * dGdt + self.basalinsulin - self.gamma * deltaIpest_k
                 

            # Update internal loop control actions
            self.uk_hist[i] = uk
            self.iob_hist[i] = self.iob
            self.gref_hist[i] = Gref_filteredk
            self.w_hist[i] = wk
            self.ip_hist[i] = deltaIpest_k

            self.u_km1 = uk
            self._gref_filtered_km1 = Gref_filteredk
            self._iob_km1 = self.iob

        # Final insulin control action to be delivered
        ukmean = max(0.0, np.mean(self.uk_hist))  # U/h
        plasma = max(0.0, np.mean(self.iob_hist))
        # Updates for next iteration
        self._glucose_km1 = cgm

        # Save controller history variables
        self._update_controller_history(insulin=ukmean, iobmax=IOBmax, kiob=KIOB)
        

        return plasma

    def estimate_iob(self, basal_insulin, bolus_insulin, controller_insulin, k=0):
        """Update current IOB estimation"""
        # Basal compartments
        C1_k = self._C1_km1 + self.dt * (max(0.0, self.basalinsulin)/60 - self._kdia * self._C1_km1)
        C2_k = self._C2_km1 + self.dt * self._kdia * (self._C1_km1 - self._C2_km1)
        self._C1_km1 = C1_k
        self._C2_km1 = C2_k

        # Deviation compartments
        deltaC1_k = self._delta_C1_km1 + self.dt * (
                max(0.0, controller_insulin)/60 + (k == 1) * bolus_insulin / self.dt -
                self.basalinsulin / 60 - self._kdia * self._delta_C1_km1)
        deltaC2_k = self._delta_C2_km1 + self.dt * self._kdia * (self._delta_C1_km1 - self._delta_C2_km1)
        self._delta_C1_km1 = deltaC1_k
        self._delta_C2_km1 = deltaC2_k
        
       
        
        # Total estimated IOB
        self.iob = C1_k + C2_k + deltaC1_k + deltaC2_k
        

    def _update_plasma_insulin(self):
        """Update estimation of plasma insulin concentration in deviation form"""

        return self._deltaIpest_km1 + self.dt * (self._kdia * self._delta_C2_km1 / self.BW / self.V_I -
                                                 self._delta_C1_km1 * self.ke)

    def _compute_bolus(self, cgm, cho_grams):
        """Computes an insulin bolus based on CGM and IOB estimation"""
        superbolus = cho_grams / self.CR + (cgm - self.Gref) / self.CF + (self.basalinsulin / 60) * cho_grams
        print("super bolus", superbolus)
        self.bolus.append(superbolus)
        return superbolus

    def _update_controller_history(self, insulin, iobmax, kiob):
        """To save history after each run_one_step execution"""
        self.history_ts['iob'].append(self.iob)
        self.history_ts['insulin'].append(insulin)
        self.history_ts['iob_max'].append(iobmax)
        self.history_ts['kiob'].append(kiob)

        self.history_dt['uk_hist'].extend(self.uk_hist)
        self.history_dt['iob_hist'].extend(self.iob_hist)
        self.history_dt['gref_hist'].extend(self.gref_hist)
        self.history_dt['w_hist'].extend(self.w_hist)
        self.history_dt['ip_hist'].extend(self.ip_hist)
    @property
    def iob(self):
        """Return insulin on board."""

        return self._iob

    @iob.setter
    def iob(self, input_iob):
        self._iob = input_iob
    


if __name__ == '__main__':
    controller = PID_SMRC_IFB(name='Patient 1', bw=70, cgm_init=100, basalinsulin=1.0, cr=10, cf=50, tdi=40)
    
#loading meal information of a specific patient
in_carbs = pd.read_excel('Datasets/carbs_RA.xlsx')
carbs_labels = pd.read_excel('Datasets/meal_labels.xlsx')
in_carbs = np.asarray(in_carbs)
carbs_labels = np.asarray(carbs_labels)

#intializing 
i=0
prev_BG = 0
mean_BG = 0
insulin_norm = 0
norm_insulin = []  #normalized insulin values
actu_insulin = []  #actual insulin values
result = []
mean_result = []
bolus = []
uk_hist=[]
bolus = []
#control loop
for x2, x3 in zip(in_carbs, carbs_labels):
    i = i+1
    print("Iteration", i)
    x1_input = array([insulin_norm]) 
    x1_input = np.expand_dims(x1_input, axis = 0)
    x2_input = array([x2])
    y_input = np.asarray(abs(randn(1,1)*(x1_input*10)))
    meal = array([x3])
    print("Meal Announcement", meal)
    # load model
    model = load_model('Two_inputs_90min4933233.h5')
    # generate BG 
    gen_BG = model.predict([x1_input, x2_input, y_input])
    gen_BG1 = gen_BG.reshape((1,18))
    transform = scaler.inverse_transform(gen_BG1)
    zeros_i = np.zeros((1,i))
    transform1 = np.concatenate((zeros_i, transform), axis=None)
    result.append(transform1)
    df = pd.DataFrame(result)
    df.replace(0, np.nan, inplace=True)
    df = df.mean(axis=0)
    mean_glu = df[i]
    insulin = controller.run_one_step(cgm=mean_glu, meal_announcement = meal, day=True)
    actu_insulin.append(insulin)
    insulin_norm = (insulin / max(in_seq1)) 
    norm_insulin.append(insulin_norm)
    
plasma = norm_insulin
df_plasma = pd.DataFrame(plasma)
plt.stem( np.arange(len(df_plasma)), plasma)
plt.title("Insulin")
plt.show()
df_plasma.to_excel('plasma_values.xlsx')
print("MEAN PI =", np.mean(norm_insulin))

bolus = np.asarray([controller.bolus]).reshape(l,1)
df_bolus = pd.DataFrame(bolus)
plt.stem( np.arange(len(df_bolus)), bolus)
plt.title("Insulin")
plt.show()
df_bolus.to_excel('bolus_values.xlsx')

df = pd.DataFrame(result)
df.replace(0, np.nan, inplace=True)
df = df.mean(axis=0)
plt.plot(df)
plt.title("Blood Glucose")
plt.show()
print("MEAN BLOOD GLUCOSE =", np.mean(df))
#stores shifted output samples to excel sheet
df.to_excel('GEN_with_cont.xlsx')
print("Total Generated Samples:", len(bolus))

 #computing glycemic metrics
average = np.mean(df)
stdd = df.std(ddof=0)
cv = (stdd/average)*100

t1=[(df>=0)&(df<=53)]
t2=[(df>=54)&(df<=69)]
t3=[(df>=70)&(df<=140)]
t4=[(df>=70)&(df<=180)]
t5=[(df>=180)&(df<=250)]
t6=[(df>=250)&(df<=400)]

a1= ((sum(t1).sum())/len(df))*100
a2= ((sum(t2).sum())/len(df))*100
a3= ((sum(t3).sum())/len(df))*100
a4= ((sum(t4).sum())/len(df))*100
a5= ((sum(t5).sum())/len(df))*100
a6= ((sum(t6).sum())/len(df))*100

print("Coefficient of variation:", round(cv,2))
print(round(a1,2))
print(round(a2,2))
print(round(a3,2))
print(round(a4,2))
print(round(a5,2))
print(round(a6,2))
