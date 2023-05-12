import streamlit as st
import pickle
import numpy as np
import sklearn



# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Laptop Price Predication")


# Brand
company = st.selectbox("Brand",df['Company'].unique())

# Type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram  = st.selectbox('Ram(in GB)',df['Ram'].unique())

# Weight
weight  = st.number_input('Weight of the laptop')

# Touchscreen
Touchscreen= st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips  = st.selectbox("IPS",['No','Yes'])

# Screen size
screen_size = st.number_input('Screen Size')

# Resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

# Cpu
cpu = st.selectbox('CPU',df['Cpu_brand'].unique())

# HDD
# hdd = st.selectbox('Hard Drive (in GB)',[0,128,256,512,1024,2048])
hdd = st.selectbox('Hard Drive (in GB)',df['HDD'].unique())

# SDD
# sdd = st.selectbox('SSD (in GB)',[0,8,128,256,512,1024])
sdd = st.selectbox('SSD (in GB)',df['SSD'].unique())

# Gpu

gpu = st.selectbox('GPU',df['Gpu_brand'].unique())

# Operating system
os = st.selectbox("operating System",df['OpSys'].unique())


if st.button("Predict Price"):
    ppi = None
    if Touchscreen=='Yes':
        Touchscreen=1
    else:
        Touchscreen = 0
    if ips =='Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi  = ((X_res**2) +(Y_res**2))**0.5/screen_size
    query = np.array([company,type,ram,os,weight,Touchscreen,ips,ppi,cpu,hdd,sdd,gpu])

    query = query.reshape(1,12)
    predication = str(int(np.exp(pipe.predict(query))[0]))
    st.title('Predication Price  : ' + predication)




