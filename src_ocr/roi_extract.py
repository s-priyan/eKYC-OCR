import pandas as pd
from src.utils import read_csv

df_dl = pd.read_csv("/home/ubuntu/home/ocr_id/template_csv/DL_info.csv" , header=None)
df_passport = pd.read_csv( "/home/ubuntu/home/ocr_id/template_csv/Passport_info.csv" , header=None )
df_new_nic = pd.read_csv("/home/ubuntu/home/ocr_id/template_csv/New_Nic_info.csv" , header=None )
df_new_nic_back = pd.read_csv("/home/ubuntu/home/ocr_id/template_csv/New_Nic_Back_info.csv" , header=None )

roi_dl = read_csv( df_dl )
roi_passport = read_csv( df_passport )
roi_new_nic = read_csv( df_new_nic )
roi_new_nic_back = read_csv( df_new_nic_back )