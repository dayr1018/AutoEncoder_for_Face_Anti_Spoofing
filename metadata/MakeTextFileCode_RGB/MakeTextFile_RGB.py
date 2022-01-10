Train_File = 'train_data_list.txt'
Valid_File = 'valid_data_list.txt'
Test_File = 'test_data_list.txt'
Test_File_OnlyTrue = 'onlytrue_test_data_list.txt'
Test_File_OnlyFalse = 'onlyfalse_test_data_list.txt'

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
            
with open(Train_File, 'a') as file:
    for fileNum in range(TrainData_Start,TrainData_End):
        for jpgNum in range(1,31):            
            file.write("Training/" + str(fileNum) + "/KINECT/Light_02_Mid/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )
            
with open(Train_File, 'a') as file:
    for fileNum in range(TrainData_Start,TrainData_End):
        for jpgNum in range(1,31):            
            file.write("Training/" + str(fileNum) + "/KINECT/Light_03_Low/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )

# Valid Data
with open(Valid_File, 'w') as file:
    for fileNum in range(ValidData_start,ValidData_End):
        for jpgNum in range(1,31):  
            file.write("Training/" + str(fileNum) + "/KINECT/Light_01_High/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg 0\n" )
            
with open(Valid_File, 'a') as file:
    for fileNum in range(ValidData_start,ValidData_End):
        for jpgNum in range(1,31):           
            file.write("Training/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg 0\n" )

with open(Valid_File, 'a') as file:
    for fileNum in range(ValidData_start,ValidData_End):
        for jpgNum in range(1,31):  
            file.write("Training/" + str(fileNum) + "/KINECT/Light_03_Low/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg 0\n" )

with open(Valid_File, 'a') as file:
    for fileNum in range(ValidData_start,ValidData_End):
        for jpgNum in range(1,31):  
            file.write("Training/" + str(fileNum) + "/KINECT/Light_01_High/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )
            
with open(Valid_File, 'a') as file:
    for fileNum in range(ValidData_start,ValidData_End):
        for jpgNum in range(1,31):           
            file.write("Training/" + str(fileNum) + "/KINECT/Light_02_Mid/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )

with open(Valid_File, 'a') as file:
    for fileNum in range(ValidData_start,ValidData_End):
        for jpgNum in range(1,31):  
            file.write("Training/" + str(fileNum) + "/KINECT/Light_03_Low/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )


# Test Data (True + False)
with open(Test_File, 'w') as file:
    for fileNum in range(TestData_Start,TestData_End):
        for jpgNum in range(1,31):  
            file.write("Test/" + str(fileNum) + "/KINECT/Light_01_High/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg 0\n" )
            
with open(Test_File, 'a') as file:
    for fileNum in range(TestData_Start,TestData_End):
        for jpgNum in range(1,31):           
            file.write("Test/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg 0\n" )

with open(Test_File, 'a') as file:
    for fileNum in range(TestData_Start,TestData_End):
        for jpgNum in range(1,31):  
            file.write("Test/" + str(fileNum) + "/KINECT/Light_03_Low/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg 0\n" )

with open(Test_File, 'a') as file:
    for fileNum in range(TestData_Start,TestData_End):
        for jpgNum in range(1,31):  
            file.write("Test/" + str(fileNum) + "/KINECT/Light_01_High/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )
            
with open(Test_File, 'a') as file:
    for fileNum in range(TestData_Start,TestData_End):
        for jpgNum in range(1,31):           
            file.write("Test/" + str(fileNum) + "/KINECT/Light_02_Mid/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )

with open(Test_File, 'a') as file:
    for fileNum in range(TestData_Start,TestData_End):
        for jpgNum in range(1,31):  
            file.write("Test/" + str(fileNum) + "/KINECT/Light_03_Low/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )


# Test Data (Only True)
with open(Test_File_OnlyTrue, 'w') as file:
    for fileNum in range(TestData_Start,TestData_End):
        for jpgNum in range(1,31):  
            file.write("Test/" + str(fileNum) + "/KINECT/Light_01_High/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )
            
with open(Test_File_OnlyTrue, 'a') as file:
    for fileNum in range(TestData_Start,TestData_End):
        for jpgNum in range(1,31):           
            file.write("Test/" + str(fileNum) + "/KINECT/Light_02_Mid/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )

with open(Test_File_OnlyTrue, 'a') as file:
    for fileNum in range(TestData_Start,TestData_End):
        for jpgNum in range(1,31):  
            file.write("Test/" + str(fileNum) + "/KINECT/Light_03_Low/real_01/color/crop/" + convert(jpgNum) + ".jpg 1\n" )


# Test Data (Only False)
with open(Test_File_OnlyFalse, 'w') as file:
    for fileNum in range(TestData_Start,TestData_End):
        for jpgNum in range(1,31):  
            file.write("Test/" + str(fileNum) + "/KINECT/Light_01_High/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg 0\n" )
            
with open(Test_File_OnlyFalse, 'a') as file:
    for fileNum in range(TestData_Start,TestData_End):
        for jpgNum in range(1,31):           
            file.write("Test/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg 0\n" )

with open(Test_File_OnlyFalse, 'a') as file:
    for fileNum in range(TestData_Start,TestData_End):
        for jpgNum in range(1,31):  
            file.write("Test/" + str(fileNum) + "/KINECT/Light_03_Low/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg 0\n" )


# Data Count 

file = open(Train_File, 'r')
print(Train_File)
print(file.read().count("\n")+1)
file.close()

file = open(Valid_File, 'r')
print(Valid_File)
print(file.read().count("\n")+1)
file.close()

file = open(Test_File, 'r')
print(Test_File)
print(file.read().count("\n")+1)
file.close()

file = open(Test_File_OnlyTrue, 'r')
print(Test_File_OnlyTrue)
print(file.read().count("\n")+1)
file.close()

file = open(Test_File_OnlyFalse, 'r')
print(Test_File_OnlyFalse)
print(file.read().count("\n")+1)
file.close()
