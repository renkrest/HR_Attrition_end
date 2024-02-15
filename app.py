# Mengimpor library
import pandas as pd
import streamlit as st
import pickle

# Menghilangkan warning
import warnings
warnings.filterwarnings("ignore")

# Menulis judul
st.markdown("<h1 style='text-align: center; '> Bank Loan Prediction </h1>", unsafe_allow_html=True)
st.markdown('---'*10)

# Load model
my_model = pickle.load(open('model_klasifikasi_bank_loan.pkl', 'rb'))

# Pilihan utama

pilihan = st.selectbox('Apa yang ingin Anda lakukan?',['Prediksi dari file csv','Input Manual'])

if pilihan == 'Prediksi dari file csv':
    # Mengupload file
    upload_file = st.file_uploader('Pilih file csv', type='csv')
    if upload_file is not None:
        dataku = pd.read_csv(upload_file)
        st.write(dataku)
        st.success('File berhasil diupload')
        hasil = my_model.predict(dataku)
        #st.write('Prediksi',hasil)
        # Keputusan
        for i in range(len(hasil)):
            if hasil[i] == 1:
                st.write('Data pelanggan',dataku['Loan ID'][i],'= diprediksi akan GAGAL BAYAR')
            else:
                st.write('Data pelanggan',dataku['Loan ID'][i],'= diprediksi akan DAPAT MEMBAYAR')
    else:
        st.error('File yang diupload kosong, silakan pilih file yang valid')
        #st.markdown('File yang diupload kosong, silakan pilih file yang valid')
else:
   # Baris Pertama
   with st.container():
       col1, col2 = st.columns(2)
       with col1:
           loan_id = st.number_input('Loan ID', value=1012)
       with col2:
           customer_id = st.number_input('Customer ID', value=1012)

           
   # Baris Kedua
   with st.container():
       col1, col2 = st.columns(2)
       with col1:
           term = st.selectbox('Term',['Short Term','Long Term'])
       with col2:
           years_in_current_job = st.selectbox('Years in current job',['< 1 year', '1 year',
                                                                       '2 years', '3 years',
                                                                       '4 years', '5 years',
                                                                       '6 years', '7 years',
                                                                       '8 years', '9 years',
                                                                       '10+ years'])  

   # Baris Ketiga
   with st.container():
       col1, col2, col3 = st.columns(3)
       with col1:
           home_ownership = st.selectbox('Home Ownership',['Home Mortgage','Own Home', 'Rent', 'HaveMortgage'])
       with col2:
           purpose = st.selectbox('Purpose',['Home Improvements','Debt Consolidation', 
                                             'Buy House', 'other',
                                             'Business Loan', 'Buy a Car',
                                             'major_purchase', 'Take a Trip',
                                             'Other', 'small_business',
                                             'Medical Bills', 'wedding',
                                             'vacation', 'Educational Expenses',
                                             'moving', 'renewable_energy'])        
       with col3:
           bankruptcies = st.selectbox('Bankruptcies',[0,1,2,3,4,5,6,7])    
           
   
            
            
   # Baris Keempat
   with st.container():
       col1, col2, col3 = st.columns(3)
       with col1:
            current_loan_amount = st.number_input('Current Loan Amount', value=400000.0)
       with col2:
            credit_score = st.number_input('Credit Score', value=750.0) 
       with col3:
            annual_income = st.number_input('Annual Income', value=1000000.0)
       
   # Baris Kelima
   with st.container():
       col1, col2, col3 = st.columns(3)
       with col1:
            monthly_debt = st.number_input('Monthly Debt', value=10000.0)
       with col2:
            years_of_credit_history = st.number_input('Years of Credit History', value=15.2) 
       with col3:
            months_since_last = st.number_input('Months since last delinquent', value=8.0)

   # Baris Keenam
   with st.container():
       col1, col2, col3 = st.columns(3)
       with col1:
            number_of_open_accounts = st.number_input('Number of Open Accounts', value=25.0)
       with col2:
            number_of_credit_problems = st.number_input('Number of Credit Problems', value=1) 
       with col3:
            current_credit_balance = st.number_input('Current Credit Balance', value=200000.0)

   # Baris Kelima
   with st.container():
       col1, col2 = st.columns(2)
       with col1:
            maximum_open_credit = st.number_input('Maximum Open Credit', value=350000.0)
       with col2:
            tax_liens = st.number_input('Tax Liens', value=1) 


   # Inference
   data = {
           'Loan ID': loan_id,
           'Customer ID': customer_id,
           'Term': term,
           'Years in current job': years_in_current_job,
           'Home Ownership': home_ownership,
           'Purpose': purpose,
           'Bankruptcies': bankruptcies,
           'Current Loan Amount': current_loan_amount, 
           'Credit Score': credit_score,
           'Annual Income': annual_income,
           'Monthly Debt': monthly_debt,
           'Years of Credit History': years_of_credit_history,
           'Months since last delinquent': months_since_last,
           'Number of Open Accounts': number_of_open_accounts,
           'Number of Credit Problems': number_of_credit_problems,
           'Current Credit Balance': current_credit_balance,
           'Maximum Open Credit': maximum_open_credit,
           'Tax Liens': tax_liens        
           }

   # Tabel data
   kolom = list(data.keys())
   df = pd.DataFrame([data.values()], columns=kolom)
   
   # Melakukan prediksi
   hasil = my_model.predict(df)
   hasil_proba = my_model.predict_proba(df)
   keputusan1 = round(float(hasil_proba[:,0])*100,2)
   keputusan2 = round(float(hasil_proba[:,1])*100,2)


   # Memunculkan hasil di Web 
   st.write('***'*10)
   st.write('<center><b><i><u><h3>Customer Loan ID', str(loan_id),'</b></i></u></h3>', unsafe_allow_html=True)
   st.write('<center><b><h4>Probabilitas bisa membayar = ', str(keputusan1),'%</b></h4>', unsafe_allow_html=True)
   st.write('<center><b><h4>Probabilitas gagal bayar = ', str(keputusan2),'%</b></h4>', unsafe_allow_html=True)