Train_File = 'train_data_list.txt'
Train_File_wo_low = 'train_data_list_wo_low.txt'

Valid_File = 'valid_data_list.txt'
Valid_File_wo_low = 'valid_data_list_wo_low.txt'
Valid_File_w_etc = 'valid_data_list_w_etc.txt' 
Valid_File_w_etc_wo_low = 'valid_data_list_w_etc_wo_low.txt' 

Test_File = 'test_data_list.txt'
Test_File_wo_low = 'test_data_list_wo_low.txt'
Test_File_w_etc = 'test_data_list_w_etc.txt' 
Test_File_w_etc_wo_low = 'test_data_list_w_etc_wo_low.txt' 

TrainData_Start = 1
TrainData_End = 28 # means 27
ValidData_start = 28
ValidData_End = 37 # means 36
TestData_Start = 37
TestData_End = 46 # means 45

def convert(num):
    if len(str(num)) == 1:
        return "00"+str(num)
    elif len(str(num)) == 2:
        return "0"+str(num)\

# Train Data
with open(Train_File, 'w') as file:
    for fileNum in range(TrainData_Start,TrainData_End):
        for jpgNum in range(1,31):
            file.write("Training/" + str(fileNum) + "/KINECT/Light_01_High/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )          
            file.write("Training/" + str(fileNum) + "/KINECT/Light_02_Mid/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )          
            file.write("Training/" + str(fileNum) + "/KINECT/Light_03_Low/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )

# Train Data (low 뺀 것)
with open(Train_File_wo_low, 'w') as file:
    for fileNum in range(TrainData_Start,TrainData_End):
        for jpgNum in range(1,31):
            file.write("Training/" + str(fileNum) + "/KINECT/Light_01_High/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )          
            file.write("Training/" + str(fileNum) + "/KINECT/Light_02_Mid/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )

# Valid Data - 3D 마스크만
with open(Valid_File, 'w') as file:
    for fileNum in range(ValidData_start,ValidData_End):
        for jpgNum in range(1,31):  
            file.write("Training/" + str(fileNum) + "/KINECT/Light_01_High/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )
            file.write("Training/" + str(fileNum) + "/KINECT/Light_02_Mid/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )
            file.write("Training/" + str(fileNum) + "/KINECT/Light_03_Low/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )

            file.write("Training/" + str(fileNum) + "/KINECT/Light_01_High/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg 0\n" )
            file.write("Training/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg 0\n" )
            file.write("Training/" + str(fileNum) + "/KINECT/Light_03_Low/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg 0\n" )
        
# Valid Data - 3D 마스크만 (low 뺌)
with open(Valid_File_wo_low, 'w') as file:
    for fileNum in range(ValidData_start,ValidData_End):
        for jpgNum in range(1,31):  
            file.write("Training/" + str(fileNum) + "/KINECT/Light_01_High/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" ) 
            file.write("Training/" + str(fileNum) + "/KINECT/Light_02_Mid/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )
            
            file.write("Training/" + str(fileNum) + "/KINECT/Light_01_High/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg 0\n" )
            file.write("Training/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg 0\n" ) 

# Valid Data - etc 추가
with open(Valid_File_w_etc, 'w') as file:
    for fileNum in range(ValidData_start,ValidData_End):
        for jpgNum in range(1,31):  
            file.write("Training/" + str(fileNum) + "/KINECT/Light_01_High/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )
            file.write("Training/" + str(fileNum) + "/KINECT/Light_02_Mid/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )
            file.write("Training/" + str(fileNum) + "/KINECT/Light_03_Low/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )

            file.write("Training/" + str(fileNum) + "/KINECT/Light_01_High/attack_01_print_none_flat/color/crop/" + convert(jpgNum) + ".jpg 0\n" ) 
            file.write("Training/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_01_print_none_flat/color/crop/" + convert(jpgNum) + ".jpg 0\n" )
            file.write("Training/" + str(fileNum) + "/KINECT/Light_03_Low/attack_01_print_none_flat/color/crop/" + convert(jpgNum) + ".jpg 0\n" )
            
            file.write("Training/" + str(fileNum) + "/KINECT/Light_01_High/attack_02_print_none_curved/color/crop/" + convert(jpgNum) + ".jpg 0\n" ) 
            file.write("Training/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_02_print_none_curved/color/crop/" + convert(jpgNum) + ".jpg 0\n" )
            file.write("Training/" + str(fileNum) + "/KINECT/Light_03_Low/attack_02_print_none_curved/color/crop/" + convert(jpgNum) + ".jpg 0\n" )
            
            file.write("Training/" + str(fileNum) + "/KINECT/Light_01_High/attack_03_print_eye_nose_mouth_flat/color/crop/" + convert(jpgNum) + ".jpg 0\n" ) 
            file.write("Training/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_03_print_eye_nose_mouth_flat/color/crop/" + convert(jpgNum) + ".jpg 0\n" )
            file.write("Training/" + str(fileNum) + "/KINECT/Light_03_Low/attack_03_print_eye_nose_mouth_flat/color/crop/" + convert(jpgNum) + ".jpg 0\n" )
            
            file.write("Training/" + str(fileNum) + "/KINECT/Light_01_High/attack_04_print_eye_nose_mouth_curved/color/crop/" + convert(jpgNum) + ".jpg 0\n" ) 
            file.write("Training/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_04_print_eye_nose_mouth_curved/color/crop/" + convert(jpgNum) + ".jpg 0\n" )
            file.write("Training/" + str(fileNum) + "/KINECT/Light_03_Low/attack_04_print_eye_nose_mouth_curved/color/crop/" + convert(jpgNum) + ".jpg 0\n" )
            
            file.write("Training/" + str(fileNum) + "/KINECT/Light_01_High/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg 0\n" ) 
            file.write("Training/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg 0\n" )
            file.write("Training/" + str(fileNum) + "/KINECT/Light_03_Low/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg 0\n" )

# Valid Data - etc 추가 (low 뺌) 
with open(Valid_File_w_etc_wo_low, 'w') as file:
    for fileNum in range(ValidData_start,ValidData_End):
        for jpgNum in range(1,31):  
            file.write("Training/" + str(fileNum) + "/KINECT/Light_01_High/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )
            file.write("Training/" + str(fileNum) + "/KINECT/Light_02_Mid/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )

            file.write("Training/" + str(fileNum) + "/KINECT/Light_01_High/attack_01_print_none_flat/color/crop/" + convert(jpgNum) + ".jpg 0\n" ) 
            file.write("Training/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_01_print_none_flat/color/crop/" + convert(jpgNum) + ".jpg 0\n" )
           
            file.write("Training/" + str(fileNum) + "/KINECT/Light_01_High/attack_02_print_none_curved/color/crop/" + convert(jpgNum) + ".jpg 0\n" ) 
            file.write("Training/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_02_print_none_curved/color/crop/" + convert(jpgNum) + ".jpg 0\n" )
            
            file.write("Training/" + str(fileNum) + "/KINECT/Light_01_High/attack_03_print_eye_nose_mouth_flat/color/crop/" + convert(jpgNum) + ".jpg 0\n" ) 
            file.write("Training/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_03_print_eye_nose_mouth_flat/color/crop/" + convert(jpgNum) + ".jpg 0\n" )
            
            file.write("Training/" + str(fileNum) + "/KINECT/Light_01_High/attack_04_print_eye_nose_mouth_curved/color/crop/" + convert(jpgNum) + ".jpg 0\n" ) 
            file.write("Training/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_04_print_eye_nose_mouth_curved/color/crop/" + convert(jpgNum) + ".jpg 0\n" )
           
            file.write("Training/" + str(fileNum) + "/KINECT/Light_01_High/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg 0\n" ) 
            file.write("Training/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg 0\n" )
           

# Test Data - 3D 마스크만
with open(Test_File, 'w') as file:
    for fileNum in range(TestData_Start,TestData_End):
        for jpgNum in range(1,31):  
            file.write("Test/" + str(fileNum) + "/KINECT/Light_01_High/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )     
            file.write("Test/" + str(fileNum) + "/KINECT/Light_02_Mid/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" ) 
            file.write("Test/" + str(fileNum) + "/KINECT/Light_03_Low/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )

            file.write("Test/" + str(fileNum) + "/KINECT/Light_01_High/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg 0\n" )           
            file.write("Test/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg 0\n" ) 
            file.write("Test/" + str(fileNum) + "/KINECT/Light_03_Low/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg 0\n" )


# Test Data - 3D 마스크만 (low 뺌)
with open(Test_File_wo_low, 'w') as file:
    for fileNum in range(TestData_Start,TestData_End):
        for jpgNum in range(1,31):  
            file.write("Test/" + str(fileNum) + "/KINECT/Light_01_High/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )          
            file.write("Test/" + str(fileNum) + "/KINECT/Light_02_Mid/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )

            file.write("Test/" + str(fileNum) + "/KINECT/Light_01_High/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg 0\n" )         
            file.write("Test/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg 0\n" )

# Test Data - etc 추가
with open(Test_File_w_etc, 'w') as file:
    for fileNum in range(TestData_Start,TestData_End):
        for jpgNum in range(1,31):  
            file.write("Test/" + str(fileNum) + "/KINECT/Light_01_High/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )     
            file.write("Test/" + str(fileNum) + "/KINECT/Light_02_Mid/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" ) 
            file.write("Test/" + str(fileNum) + "/KINECT/Light_03_Low/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )

            file.write("Test/" + str(fileNum) + "/KINECT/Light_01_High/attack_01_print_none_flat/color/crop/" + convert(jpgNum) + ".jpg 0\n" )           
            file.write("Test/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_01_print_none_flat/color/crop/" + convert(jpgNum) + ".jpg 0\n" ) 
            file.write("Test/" + str(fileNum) + "/KINECT/Light_03_Low/attack_01_print_none_flat/color/crop/" + convert(jpgNum) + ".jpg 0\n" )

            file.write("Test/" + str(fileNum) + "/KINECT/Light_01_High/attack_02_print_none_curved/color/crop/" + convert(jpgNum) + ".jpg 0\n" )           
            file.write("Test/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_02_print_none_curved/color/crop/" + convert(jpgNum) + ".jpg 0\n" ) 
            file.write("Test/" + str(fileNum) + "/KINECT/Light_03_Low/attack_02_print_none_curved/color/crop/" + convert(jpgNum) + ".jpg 0\n" )

            file.write("Test/" + str(fileNum) + "/KINECT/Light_01_High/attack_03_print_eye_nose_mouth_flat/color/crop/" + convert(jpgNum) + ".jpg 0\n" )           
            file.write("Test/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_03_print_eye_nose_mouth_flat/color/crop/" + convert(jpgNum) + ".jpg 0\n" ) 
            file.write("Test/" + str(fileNum) + "/KINECT/Light_03_Low/attack_03_print_eye_nose_mouth_flat/color/crop/" + convert(jpgNum) + ".jpg 0\n" )

            file.write("Test/" + str(fileNum) + "/KINECT/Light_01_High/attack_04_print_eye_nose_mouth_curved/color/crop/" + convert(jpgNum) + ".jpg 0\n" )           
            file.write("Test/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_04_print_eye_nose_mouth_curved/color/crop/" + convert(jpgNum) + ".jpg 0\n" ) 
            file.write("Test/" + str(fileNum) + "/KINECT/Light_03_Low/attack_04_print_eye_nose_mouth_curved/color/crop/" + convert(jpgNum) + ".jpg 0\n" )

            file.write("Test/" + str(fileNum) + "/KINECT/Light_01_High/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg 0\n" )           
            file.write("Test/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg 0\n" ) 
            file.write("Test/" + str(fileNum) + "/KINECT/Light_03_Low/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg 0\n" )

# Test Data - etc 추가 (low 뺌)
with open(Test_File_w_etc_wo_low, 'w') as file:
    for fileNum in range(TestData_Start,TestData_End):
        for jpgNum in range(1,31):  
            file.write("Test/" + str(fileNum) + "/KINECT/Light_01_High/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )     
            file.write("Test/" + str(fileNum) + "/KINECT/Light_02_Mid/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" ) 

            file.write("Test/" + str(fileNum) + "/KINECT/Light_01_High/attack_01_print_none_flat/color/crop/" + convert(jpgNum) + ".jpg 0\n" )           
            file.write("Test/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_01_print_none_flat/color/crop/" + convert(jpgNum) + ".jpg 0\n" ) 

            file.write("Test/" + str(fileNum) + "/KINECT/Light_01_High/attack_02_print_none_curved/color/crop/" + convert(jpgNum) + ".jpg 0\n" )           
            file.write("Test/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_02_print_none_curved/color/crop/" + convert(jpgNum) + ".jpg 0\n" ) 

            file.write("Test/" + str(fileNum) + "/KINECT/Light_01_High/attack_03_print_eye_nose_mouth_flat/color/crop/" + convert(jpgNum) + ".jpg 0\n" )           
            file.write("Test/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_03_print_eye_nose_mouth_flat/color/crop/" + convert(jpgNum) + ".jpg 0\n" ) 
        
            file.write("Test/" + str(fileNum) + "/KINECT/Light_01_High/attack_04_print_eye_nose_mouth_curved/color/crop/" + convert(jpgNum) + ".jpg 0\n" )           
            file.write("Test/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_04_print_eye_nose_mouth_curved/color/crop/" + convert(jpgNum) + ".jpg 0\n" ) 
           
            file.write("Test/" + str(fileNum) + "/KINECT/Light_01_High/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg 0\n" )           
            file.write("Test/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg 0\n" ) 

files = [
            Train_File, Train_File_wo_low, 
            Valid_File, Valid_File_wo_low, Valid_File_w_etc, Valid_File_w_etc_wo_low, 
            Test_File, Test_File_wo_low, Test_File_w_etc, Test_File_w_etc_wo_low 
        ]

# Data Count 
for filename in files:
    file = open(filename, 'r')
    print(filename)
    print(file.read().count("\n")+1)
    file.close()
