# SAC-LSTM

This is a MindSpore implementation of SAC-LSTM, a recurrent model for radar echo extrapolation (precipitation nowcasting). Our implementation code in Pytorch is available at https://github.com/LeiShe1/SAC-LSTM.



# Setup

Required python libraries: MindSpore==1.8.0 cann==5.1.2 python==3.7 


# Datasets

We conduct experiments on CIKM AnalytiCup 2017 datasets: [CIKM_AnalytiCup_Address](https://tianchi.aliyun.com/competition/entrance/231596/information) or [CIKM_Rardar](https://drive.google.com/drive/folders/1IqQyI8hTtsBbrZRRht3Es9eES_S4Qv2Y?usp=sharing) 

# Training

Use 'SAC_shengteng.py' script to train these models. 

You might want to change the parameter and setting, you can change the details of variable ‘args’ 


# Test
Use 'SAC_shengteng_test.py' script to test these models. 


