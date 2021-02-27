import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
# from numpy import asarray
import pandas as pd
import requests
import json
import numpy as np
from entsoe import EntsoePandasClient
import statsmodels.api as sm


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def lower_confidence_interval(maindata, prediction):
    s = np.std(maindata) # std of vector
    z = 1.96 # for a 95% CI
    lower = prediction - (z * s)
    return lower

def upper_confidence_interval(maindata, prediction):
    s = np.std(maindata) # std of vector
    z = 1.96 # for a 95% CI
    upper = prediction + (z * s)
    return upper

#%%%###API#############
def EPIAS_API():
    down = './test.json'
    url = 'https://seffaflik.epias.com.tr/transparency/service/market/day-ahead-mcp?endDate=2019-12-31&startDate=2017-01-01'
    outpath=down
    generatedURL=url
    response = requests.get(generatedURL)
    if response.status_code == 200:
        with open(outpath, "wb") as out:
            for chunk in response.iter_content(chunk_size=128):
                out.write(chunk)
    with open(down) as json_file:
        data = json.load(json_file)
    body=data.get('body')
    gen=body.get('dayAheadMCPList')
    df=pd.DataFrame(gen)
    return(df)

#%%#############ENTSOE-API####################
def ENTSOE_API():
    client = EntsoePandasClient(api_key="2c958a88-3776-4f01-82cd-c957fdc4dc6a")

    country_code = 'EE', 'PT', 'ES', 'FR', 'FI', 'HU', 'SI', 'LV', 'NL', 'GR', 'BE'

    start = [pd.Timestamp('2016-12-31T22:00Z'), pd.Timestamp('2016-12-31T23:00Z'), pd.Timestamp('2016-12-31T23:00Z'), pd.Timestamp('2016-12-31T23:00Z'), pd.Timestamp('2016-12-31T22:00Z'), pd.Timestamp('2016-12-31T23:00Z'), pd.Timestamp('2016-12-31T23:00Z'), pd.Timestamp('2016-12-31T22:00Z'), pd.Timestamp('2016-12-31T23:00Z'), pd.Timestamp('2016-12-31T22:00Z'), pd.Timestamp('2016-12-31T23:00Z')]
    end= [pd.Timestamp('2019-12-31T21:00Z'), pd.Timestamp('2019-12-31T22:00Z'), pd.Timestamp('2019-12-31T22:00Z'), pd.Timestamp('2019-12-31T22:00Z'), pd.Timestamp('2019-12-31T21:00Z'), pd.Timestamp('2019-12-31T23:00Z'), pd.Timestamp('2019-12-31T23:00Z'), pd.Timestamp('2019-12-31T21:00Z'), pd.Timestamp('2019-12-31T23:00Z'), pd.Timestamp('2019-12-31T21:00Z'), pd.Timestamp('2019-12-31T22:00Z')]

    df1=[]
    iteration2=0
    ElectricityPrice=[]
    for iiii in range(len(country_code)):
        ElectricityPrice=client.query_day_ahead_prices(country_code[iteration2], start=start[iteration2], end=end[iteration2])
        if iiii==0:
            df1=pd.DataFrame({country_code[iteration2]:ElectricityPrice.values})
            iteration2=iteration2+1
            print(df1)
        else:
            df1[country_code[iteration2]]=pd.DataFrame({country_code[iteration2]:ElectricityPrice.values})
            iteration2=iteration2+1
            print(df1)
    return(df1)
#%%############   SPLIT TRAIN TEST MODEL   #############
df = ENTSOE_API()
df1 = EPIAS_API()
df1 = df1['priceEur']

countrycode = 'TR'
df[countrycode]=pd.DataFrame({countrycode:df1.values})
#%%####################################################

#############       PREPROCESSING        ##############

#######################################################

newcountrycode = 'EE', 'PT', 'ES', 'FR', 'FI', 'HU', 'SI', 'LV', 'NL', 'GR', 'BE', 'TR'

fig, ax = plt.subplots(3, 4, figsize=(25,15))
iter1=0
iteration4=0
iteration5=0
RMSEsmoothingday = []
RMSEsmoothingweek = []
RMSEholtsDay = []
RMSEholtsweek = []
RMSENaiveMethodday = []
RMSENaiveMethodweek = []
RMSEWinterDay = []
RMSEWinterWeek = []

######
Tahminsmoothingday = pd.DataFrame(columns=df.columns).fillna(0)
lowersmoothingday = pd.DataFrame(columns=df.columns).fillna(0)
uppersmoothingday = pd.DataFrame(columns=df.columns).fillna(0)
Tahminsmoothingweek = pd.DataFrame(columns=df.columns).fillna(0)
lowersmoothingweek = pd.DataFrame(columns=df.columns).fillna(0)
uppersmoothingweek = pd.DataFrame(columns=df.columns).fillna(0)
######
TahminholtsDay = pd.DataFrame(columns=df.columns).fillna(0)
lowerholtsDay = pd.DataFrame(columns=df.columns).fillna(0)
upperholtsDay = pd.DataFrame(columns=df.columns).fillna(0)
Tahminholtsweek = pd.DataFrame(columns=df.columns).fillna(0)
lowerholtsweek = pd.DataFrame(columns=df.columns).fillna(0)
upperholtsweek = pd.DataFrame(columns=df.columns).fillna(0)
######
TahminNaiveday = pd.DataFrame(columns=df.columns).fillna(0)
lowerNaiveday = pd.DataFrame(columns=df.columns).fillna(0)
upperNaiveday = pd.DataFrame(columns=df.columns).fillna(0)
TahminNaiveweek = pd.DataFrame(columns=df.columns).fillna(0)
lowerNaiveweek = pd.DataFrame(columns=df.columns).fillna(0)
upperNaiveweek = pd.DataFrame(columns=df.columns).fillna(0)
######
TahminWinterDay = pd.DataFrame(columns=df.columns).fillna(0)
lowerWinterDay = pd.DataFrame(columns=df.columns).fillna(0)
upperWinterDay = pd.DataFrame(columns=df.columns).fillna(0)
TahminWinterweek = pd.DataFrame(columns=df.columns).fillna(0)
lowerWinterweek = pd.DataFrame(columns=df.columns).fillna(0)
upperWinterweek = pd.DataFrame(columns=df.columns).fillna(0)
######

for iterationnew in range(len(newcountrycode)):
    PTF20152019 = df[newcountrycode[iterationnew]]
    
#%%############################################################

############    SINGLE EXPONENTIAL SMOOTHING       ############

###############################################################

############ Day #############    
    k=0.1
    Validationsmoothingday = np.zeros((len(PTF20152019), 9))
    for iteration in range(9):
        initialvalue=PTF20152019[0]
        Validationsmoothingday[24][iteration] = initialvalue
        i=24
        while i < len(PTF20152019): #%% not equal bcs dont forecast a new day.
            initialvalue=(k*PTF20152019[i-24])+((1-k)*initialvalue)
            Validationsmoothingday[i][iteration] = initialvalue
            i=i+1
        k=k+0.1
        
    Validationsmoothingday = pd.DataFrame(Validationsmoothingday)
    Tahminsmoothingday[newcountrycode[iterationnew]] = Validationsmoothingday[8][26112:26280]
    lowersmoothingday[newcountrycode[iterationnew]] = lower_confidence_interval(PTF20152019, Tahminsmoothingday[newcountrycode[iterationnew]])
    uppersmoothingday[newcountrycode[iterationnew]] = upper_confidence_interval(PTF20152019, Tahminsmoothingday[newcountrycode[iterationnew]])
    
    # RMSEsmoothingday.append(rmse(Validationsmoothingday.loc[:,8][26256:26280], PTF20152019[26256:26280]))

########### Week ###########
    k=0.1
    Validationsmoothingweek = np.zeros((len(PTF20152019), 9))
    for iteration in range(9):
        initialvalue=PTF20152019[0]
        Validationsmoothingweek[168][iteration] = initialvalue
        i=168;
        while i < len(PTF20152019): ## not equal bcs dont forecast a new day.
            initialvalue=(k*PTF20152019[i-168])+((1-k)*initialvalue)
            Validationsmoothingweek[i][iteration]=initialvalue
            i=i+1
        k=k+0.1
        
    Validationsmoothingweek = pd.DataFrame(Validationsmoothingweek)
    Tahminsmoothingweek[newcountrycode[iterationnew]] = Validationsmoothingweek[8][26112:26280]
    lowersmoothingweek[newcountrycode[iterationnew]] = lower_confidence_interval(PTF20152019, Tahminsmoothingweek[newcountrycode[iterationnew]])
    uppersmoothingweek[newcountrycode[iterationnew]] = upper_confidence_interval(PTF20152019, Tahminsmoothingweek[newcountrycode[iterationnew]])
    
    # RMSEsmoothingweek.append(rmse(Validationsmoothingweek.loc[:,8][26112:26280], PTF20152019[26112:26280]))
    
#%%##################################################################

############ (DOUBLE EXPONENTIAL SMOOTHING) HOLTS METHOD ############

#####################################################################

############ Day #############
    k=0.1
    Lvalue = np.zeros((len(PTF20152019), 9))
    Tvalue = np.zeros((len(PTF20152019), 9))
    ValidationholtsDay = np.zeros((len(PTF20152019), 9))
    for iteration in range(9):
        p=1;
        beta=0.01
        Lvalue_initial=PTF20152019[0]
        Tvalue_initial=0
        #initial forecast for i=25
        Lvalue[0][iteration]=k*PTF20152019[0]+(1-k)*(Lvalue_initial+Tvalue_initial)
        Tvalue[0][iteration]=beta*(Lvalue[0][iteration]-Lvalue_initial)+((1-beta)*Tvalue_initial)
        ValidationholtsDay[24][iteration]=Lvalue[0][iteration]+p*Tvalue[0][iteration]
    
        i=25;
        while i < len(PTF20152019): 
            Lvalue[i-24][iteration]=k*PTF20152019[i-24]+(1-k)*(Lvalue[i-25][iteration]+Tvalue[i-25][iteration])
            Tvalue[i-24][iteration]=beta*(Lvalue[i-24][iteration]-Lvalue[i-25][iteration])+((1-beta)*Tvalue[i-25][iteration])
            ValidationholtsDay[i][iteration]=Lvalue[i-24][iteration]+p*Tvalue[i-24][iteration];
            i=i+1
        k=k+0.1

    ValidationholtsDay = pd.DataFrame(ValidationholtsDay)
    
    TahminholtsDay[newcountrycode[iterationnew]] = ValidationholtsDay[8][26112:26280]
    lowerholtsDay[newcountrycode[iterationnew]] = lower_confidence_interval(PTF20152019, TahminholtsDay[newcountrycode[iterationnew]])
    upperholtsDay[newcountrycode[iterationnew]] = upper_confidence_interval(PTF20152019, TahminholtsDay[newcountrycode[iterationnew]])
    
    # RMSEholtsDay.append(rmse(ValidationholtsDay.loc[:,8][26256:26280], PTF20152019[26256:26280]))

########### Week ###########
    k=0.1
    Lvalue = np.zeros((len(PTF20152019), 9))
    Tvalue = np.zeros((len(PTF20152019), 9))
    ValidationholtsWeek = np.zeros((len(PTF20152019), 9))
    for iteration in range(9):
        p=1;
        beta=0.01
        Lvalue_initial=PTF20152019[0]
        Tvalue_initial=0
        #initial forecast for i=170
        Lvalue[0][iteration]=k*PTF20152019[0]+(1-k)*(Lvalue_initial+Tvalue_initial)
        Tvalue[0][iteration]=beta*(Lvalue[0][iteration]-Lvalue_initial)+((1-beta)*Tvalue_initial)
        ValidationholtsWeek[168][iteration]=Lvalue[0][iteration]+p*Tvalue[0][iteration]
    
        i=169;
        while i < len(PTF20152019): 
            Lvalue[i-168][iteration]=k*PTF20152019[i-168]+(1-k)*(Lvalue[i-169][iteration]+Tvalue[i-169][iteration])
            Tvalue[i-168][iteration]=beta*(Lvalue[i-168][iteration]-Lvalue[i-169][iteration])+((1-beta)*Tvalue[i-169][iteration])
            ValidationholtsWeek[i][iteration]=Lvalue[i-168][iteration]+p*Tvalue[i-168][iteration]
            i=i+1
        k=k+0.1
        
    ValidationholtsWeek = pd.DataFrame(ValidationholtsWeek)
    
    Tahminholtsweek[newcountrycode[iterationnew]] = ValidationholtsWeek[8][26112:26280]
    lowerholtsweek[newcountrycode[iterationnew]] = lower_confidence_interval(PTF20152019, Tahminholtsweek[newcountrycode[iterationnew]])
    upperholtsweek[newcountrycode[iterationnew]] = upper_confidence_interval(PTF20152019, Tahminholtsweek[newcountrycode[iterationnew]])
    
    # RMSEholtsweek.append(rmse(ValidationholtsWeek.loc[:,8][26112:26280], PTF20152019[26112:26280]))
    
#%%#####################################################

################### NAIVE METHOD #######################

########################################################

#### Day ####
    i=24
    NaiveMethodday = np.zeros(len(PTF20152019))
    while i < len(PTF20152019):
        NaiveMethodday[i]=PTF20152019[i-24]
        i=i+1

    NaiveMethodday = pd.DataFrame(NaiveMethodday)
    
    TahminNaiveday[newcountrycode[iterationnew]] = NaiveMethodday[0][26112:26280]
    lowerNaiveday[newcountrycode[iterationnew]] = lower_confidence_interval(PTF20152019, TahminNaiveday[newcountrycode[iterationnew]])
    upperNaiveday[newcountrycode[iterationnew]] = upper_confidence_interval(PTF20152019, TahminNaiveday[newcountrycode[iterationnew]])
    
    # RMSENaiveMethodday.append(rmse(NaiveMethodday[0][26256:26280], PTF20152019[26256:26280]))

#### Week ####
    i=168;
    NaiveMethodweek = np.zeros(len(PTF20152019))
    while i < len(PTF20152019):
        NaiveMethodweek[i]=PTF20152019[i-168]
        i=i+1;

    NaiveMethodweek = pd.DataFrame(NaiveMethodweek)
    
    TahminNaiveweek[newcountrycode[iterationnew]] = NaiveMethodweek[0][26112:26280]
    lowerNaiveweek[newcountrycode[iterationnew]] = lower_confidence_interval(PTF20152019, TahminNaiveweek[newcountrycode[iterationnew]])
    upperNaiveweek[newcountrycode[iterationnew]] = upper_confidence_interval(PTF20152019, TahminNaiveweek[newcountrycode[iterationnew]])
    
    # RMSENaiveMethodweek.append(rmse(NaiveMethodweek[0][26112:26280], PTF20152019[26112:26280]))
    
#%%############################################################

#################### HOLT'S WINTER METHOD #####################

###############################################################

##### Day #####
    p=1;
    k=0.1
    Lvalue = np.zeros((len(PTF20152019), 9))
    Tvalue = np.zeros((len(PTF20152019), 9))
    Svalue = np.zeros((len(PTF20152019), 9))
    ValidationWinterDay = np.zeros((len(PTF20152019), 9))
    for iteration in range(9):
        beta=0.01;
        gamma = 0.9;
        Lvalue_initial=PTF20152019[0]
        Tvalue_initial=PTF20152019[1]-PTF20152019[0]
        Svalue_initial=PTF20152019[0]/Lvalue_initial;
        #initial forecast for i=169
        Lvalue[0][iteration]=k*(PTF20152019[0]/Svalue_initial)+(1-k)*(Lvalue_initial+Tvalue_initial)
        Tvalue[0][iteration]=beta*(Lvalue[0][iteration]-Lvalue_initial)+((1-beta)*Tvalue_initial)
        Svalue[0][iteration]=gamma*(PTF20152019[0]/Lvalue_initial)+(1-gamma)*Svalue_initial
        ValidationWinterDay[24][iteration]=(Lvalue[0][iteration]+p*Tvalue[0][iteration])*Svalue[0][iteration]
        
        i=25
        while i < len(PTF20152019):
            Lvalue[i-24][iteration] = k * PTF20152019[i-24] + (1-k) * (Lvalue[i-25][iteration] + Tvalue[i-25][iteration])
            Tvalue[i-24][iteration] = beta * (Lvalue[i-24][iteration]-Lvalue[i-25][iteration])+((1-beta)*Tvalue[i-25][iteration])
            Svalue[i-24][iteration] = gamma * (PTF20152019[i-24]/Lvalue[i-24][iteration]) + ((1-gamma)*Svalue[i-25][iteration])
            ValidationWinterDay[i][iteration] = (Lvalue[i-24][iteration]+p*Tvalue[i-24][iteration])*Svalue[i-24][iteration];
            i=i+1
        k=k+0.1

    ValidationWinterDay = pd.DataFrame(ValidationWinterDay)
    
    TahminWinterDay[newcountrycode[iterationnew]] = ValidationWinterDay[8][26112:26280]
    lowerWinterDay[newcountrycode[iterationnew]] = lower_confidence_interval(PTF20152019, TahminWinterDay[newcountrycode[iterationnew]])
    upperWinterDay[newcountrycode[iterationnew]] = upper_confidence_interval(PTF20152019, TahminWinterDay[newcountrycode[iterationnew]])
    
    # RMSEWinterDay.append(rmse(ValidationWinterDay.loc[:,8][26256:26280], PTF20152019[26256:26280]))

########## Week ##########
    p=1;
    k=0.1
    Lvalue = np.zeros((len(PTF20152019), 9))
    Tvalue = np.zeros((len(PTF20152019), 9))
    Svalue = np.zeros((len(PTF20152019), 9))
    ValidationWinterWeek = np.zeros((len(PTF20152019), 9))
    for iteration in range(9):
        beta=0.01;
        gamma = 0.9;
        Lvalue_initial=PTF20152019[0]
        Tvalue_initial=PTF20152019[1]-PTF20152019[0]
        Svalue_initial=PTF20152019[0]/Lvalue_initial;
        #initial forecast for i=169
        Lvalue[0][iteration]=k*(PTF20152019[0]/Svalue_initial)+(1-k)*(Lvalue_initial+Tvalue_initial)
        Tvalue[0][iteration]=beta*(Lvalue[0][iteration]-Lvalue_initial)+((1-beta)*Tvalue_initial)
        Svalue[0][iteration]=gamma*(PTF20152019[0]/Lvalue_initial)+(1-gamma)*Svalue_initial
        ValidationWinterWeek[168][iteration]=(Lvalue[0][iteration]+p*Tvalue[0][iteration])*Svalue[0][iteration]
      
        i=169
        while i < len(PTF20152019):
            Lvalue[i-168][iteration] = k * PTF20152019[i-168] + (1-k) * (Lvalue[i-169][iteration] + Tvalue[i-169][iteration])
            Tvalue[i-168][iteration] = beta * (Lvalue[i-168][iteration]-Lvalue[i-169][iteration])+((1-beta)*Tvalue[i-169][iteration])
            Svalue[i-168][iteration] = gamma * (PTF20152019[i-168]/Lvalue[i-168][iteration]) + ((1-gamma)*Svalue[i-169][iteration])
            ValidationWinterWeek[i][iteration] = (Lvalue[i-168][iteration]+(p*Tvalue[i-168][iteration])*Svalue[i-168][iteration]);
            i=i+1
        k=k+0.1

    ValidationWinterWeek = pd.DataFrame(ValidationWinterWeek)
    
    TahminWinterweek[newcountrycode[iterationnew]] = ValidationWinterWeek[8][26112:26280]
    if iterationnew == 1:
        ValidationWinterWeek[8].iloc[26190] = 1.1
        ValidationWinterWeek[8].iloc[26191] = 1.1
        TahminWinterweek['PT'].iloc[78] = 1.1
    lowerWinterweek[newcountrycode[iterationnew]] = lower_confidence_interval(PTF20152019, TahminWinterweek[newcountrycode[iterationnew]])
    upperWinterweek[newcountrycode[iterationnew]] = upper_confidence_interval(PTF20152019, TahminWinterweek[newcountrycode[iterationnew]])
    
    # RMSEWinterWeek.append(rmse(ValidationWinterWeek.loc[:,8][26112:26280], PTF20152019[26112:26280]))
#%%############################################################

######################        PLOT        #####################

###############################################################

    x = np.arange(0, 168)
    countriesnames = 'Estonia', 'Portugal', 'Spain', 'France', 'Finland', 'Hungary', 'Slovenia', 'Latvia', 'Netherlands', 'Greece', 'Belgium', 'Turkey'
    ax[iteration4, iteration5].plot(x, ValidationWinterDay[8][26112:26280], '.-', linewidth=2, color='tab:blue', label="Day Method Prices")
    ax[iteration4, iteration5].plot(x, ValidationWinterWeek[8][26112:26280], '.-', linewidth=2, color='tab:green', label="Week Method Prices")
    ax[iteration4, iteration5].plot(x, df[newcountrycode[iterationnew]][26112:26280], '.-', linewidth=2, color='tab:orange', label="Actual Prices")
    ax[iteration4, iteration5].fill_between(x, lowerWinterDay[newcountrycode[iterationnew]], upperWinterDay[newcountrycode[iterationnew]], color='dodgerblue', alpha=0.2, label="Confidence Interval (95%)")
    ax[iteration4, iteration5].fill_between(x, lowerWinterweek[newcountrycode[iterationnew]], upperWinterweek[newcountrycode[iterationnew]], color='forestgreen', alpha=0.2, label="Confidence Interval (95%)")
    ax[iteration4, iteration5].set_title(countriesnames[iter1], fontsize=20)
    ax[iteration4, iteration5].set_xlabel('Hours', fontsize=12)
    ax[iteration4, iteration5].set_ylabel('Prices (Euro/MWh)', fontsize=12)
    ax[iteration4, iteration5].legend(loc="best")
    ax[iteration4, iteration5].legend(fontsize=8)
    iter1 = iter1 + 1
    iteration5 = iteration5 + 1
    if iteration5 == 4:
        iteration4 = iteration4 + 1
        iteration5 = 0
fig.tight_layout()
plt.savefig('WinterMethod.png')
plt.savefig('WinterMethod.eps')
plt.show()
