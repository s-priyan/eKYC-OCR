import pandas as pd

def read_csv( df ):
  samples = []
  for i_row in df.iterrows() :
    name , x_min , y_min , width , height =  i_row[1][0] , i_row[1][1] , i_row[1][2] , i_row[1][3] , i_row[1][4]
    i_sample ={
        "id":name ,
        "bbox":[ int(x_min) , int(y_min) , int( x_min+width ) , int(y_min+height)   ]
    }
    samples.append(  i_sample )

  return samples

df_dl = pd.read_csv("/home/ubuntu/home/eKYC_s3/template_csv/DL_info.csv" , header=None)
df_passport = pd.read_csv( "/home/ubuntu/home/eKYC_s3/template_csv/Passport_info.csv" , header=None )
df_new_nic = pd.read_csv("/home/ubuntu/home/eKYC_s3/template_csv/New_Nic_info.csv" , header=None )
df_new_nic_back = pd.read_csv("/home/ubuntu/home/eKYC_s3/template_csv/New_Nic_Back_info.csv" , header=None )

roi_dl = read_csv( df_dl )
roi_passport = read_csv( df_passport )
roi_new_nic = read_csv( df_new_nic )
roi_new_nic_back = read_csv( df_new_nic_back )

df_sim_dl = pd.read_csv("/home/ubuntu/home/eKYC_s3/templates_similarity_csv/DL_info.csv" , header=None)
df_sim_passport = pd.read_csv( "/home/ubuntu/home/eKYC_s3/templates_similarity_csv/Passport_info.csv" , header=None )
df_sim_new_nic = pd.read_csv("/home/ubuntu/home/eKYC_s3/templates_similarity_csv/New_Nic_info.csv" , header=None )
df_sim_old_nic = pd.read_csv("/home/ubuntu/home/eKYC_s3/templates_similarity_csv/Old_Nic_info.csv" , header=None )

roi_sim_dl = read_csv( df_sim_dl )
roi_sim_passport = read_csv( df_sim_passport )
roi_sim_new_nic = read_csv( df_sim_new_nic )
roi_sim_old_nic = read_csv( df_sim_old_nic )