import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def generate_data():
    data_input=pd.read_csv('dataset/dataOfYZJPTC.csv')
    data_input.columns=['worker_No','identity','worker_place','age','sex','wage','hospital_No','hospital_grade','days','drug_fees','line','ratio','chroric_or_not','in_hospital','out_hospital','trement_fees','bed_fees','operation_fees'
    ,'care_fees','material_fees','group_fees']

    ####选取有用的列
    values = data_input[['identity','age','sex','wage','hospital_grade','days','drug_fees','line','ratio','chroric_or_not'
    ,'trement_fees','bed_fees','operation_fees','care_fees','material_fees','group_fees']]

    ##对个别列进行哑变量处理(7个星期特征)
    ####工作性质
    # worker_place = pd.get_dummies(values.worker_place, prefix='worker_place')
    # values = values.join(worker_place)
    #####在职、离退状态（2）
    identity = pd.get_dummies(values.identity, prefix='identity')
    values = values.join(identity)
    #######性别 （2）
    sex = pd.get_dummies(values.sex, prefix='sex')
    values = values.join(sex)
    #######医院等级（6）
    hospital_grade = pd.get_dummies(values.hospital_grade, prefix='hospital_grade')
    values = values.join(hospital_grade)
    #######是否患有慢性病
    chroric_or_not = pd.get_dummies(values.chroric_or_not, prefix='chroric_or_not')
    values = values.join(chroric_or_not)
    ######进行哑变量处理
    values.drop(['identity','sex','hospital_grade','chroric_or_not'],1)
    ######normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    print(scaled)
    return scaled

if __name__ == "__main__":
    generate_data()