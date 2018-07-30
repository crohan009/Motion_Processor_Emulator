################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Each subdirectory must supply rules for building sources it contributes
include/LPS25.obj: ../include/LPS25.c $(GEN_OPTS) | $(GEN_HDRS)
	@echo 'Building file: "$<"'
	@echo 'Invoking: ARM Compiler'
	"/Applications/ti/ccsv8/tools/compiler/ti-cgt-arm_18.1.1.LTS/bin/armcl" -mv7M4 --code_state=16 --float_support=FPv4SPD16 -me --include_path="/Users/crohan009/Documents/Stuff/USR18/code base/Data_extraction/i2c" --include_path="/Applications/ti/TivaWare_C_Series-2.1.4.178" --include_path="/Applications/ti/TivaWare_C_Series-2.1.4.178/utils" --include_path="/Users/crohan009/Documents/Stuff/USR18/code base/Data_extraction/i2c/include" --include_path="/Applications/ti/ccsv8/tools/compiler/ti-cgt-arm_18.1.1.LTS/include" --define=ccs="ccs" --define=PART_TM4C123GE6PM -g --gcc --diag_warning=225 --diag_wrap=off --display_error_number --abi=eabi --preproc_with_compile --preproc_dependency="include/LPS25.d_raw" --obj_directory="include" $(GEN_OPTS__FLAG) "$<"
	@echo 'Finished building: "$<"'
	@echo ' '

include/i2c_tm4c.obj: ../include/i2c_tm4c.c $(GEN_OPTS) | $(GEN_HDRS)
	@echo 'Building file: "$<"'
	@echo 'Invoking: ARM Compiler'
	"/Applications/ti/ccsv8/tools/compiler/ti-cgt-arm_18.1.1.LTS/bin/armcl" -mv7M4 --code_state=16 --float_support=FPv4SPD16 -me --include_path="/Users/crohan009/Documents/Stuff/USR18/code base/Data_extraction/i2c" --include_path="/Applications/ti/TivaWare_C_Series-2.1.4.178" --include_path="/Applications/ti/TivaWare_C_Series-2.1.4.178/utils" --include_path="/Users/crohan009/Documents/Stuff/USR18/code base/Data_extraction/i2c/include" --include_path="/Applications/ti/ccsv8/tools/compiler/ti-cgt-arm_18.1.1.LTS/include" --define=ccs="ccs" --define=PART_TM4C123GE6PM -g --gcc --diag_warning=225 --diag_wrap=off --display_error_number --abi=eabi --preproc_with_compile --preproc_dependency="include/i2c_tm4c.d_raw" --obj_directory="include" $(GEN_OPTS__FLAG) "$<"
	@echo 'Finished building: "$<"'
	@echo ' '

include/imu.obj: ../include/imu.c $(GEN_OPTS) | $(GEN_HDRS)
	@echo 'Building file: "$<"'
	@echo 'Invoking: ARM Compiler'
	"/Applications/ti/ccsv8/tools/compiler/ti-cgt-arm_18.1.1.LTS/bin/armcl" -mv7M4 --code_state=16 --float_support=FPv4SPD16 -me --include_path="/Users/crohan009/Documents/Stuff/USR18/code base/Data_extraction/i2c" --include_path="/Applications/ti/TivaWare_C_Series-2.1.4.178" --include_path="/Applications/ti/TivaWare_C_Series-2.1.4.178/utils" --include_path="/Users/crohan009/Documents/Stuff/USR18/code base/Data_extraction/i2c/include" --include_path="/Applications/ti/ccsv8/tools/compiler/ti-cgt-arm_18.1.1.LTS/include" --define=ccs="ccs" --define=PART_TM4C123GE6PM -g --gcc --diag_warning=225 --diag_wrap=off --display_error_number --abi=eabi --preproc_with_compile --preproc_dependency="include/imu.d_raw" --obj_directory="include" $(GEN_OPTS__FLAG) "$<"
	@echo 'Finished building: "$<"'
	@echo ' '

include/mpu_9150.obj: ../include/mpu_9150.c $(GEN_OPTS) | $(GEN_HDRS)
	@echo 'Building file: "$<"'
	@echo 'Invoking: ARM Compiler'
	"/Applications/ti/ccsv8/tools/compiler/ti-cgt-arm_18.1.1.LTS/bin/armcl" -mv7M4 --code_state=16 --float_support=FPv4SPD16 -me --include_path="/Users/crohan009/Documents/Stuff/USR18/code base/Data_extraction/i2c" --include_path="/Applications/ti/TivaWare_C_Series-2.1.4.178" --include_path="/Applications/ti/TivaWare_C_Series-2.1.4.178/utils" --include_path="/Users/crohan009/Documents/Stuff/USR18/code base/Data_extraction/i2c/include" --include_path="/Applications/ti/ccsv8/tools/compiler/ti-cgt-arm_18.1.1.LTS/include" --define=ccs="ccs" --define=PART_TM4C123GE6PM -g --gcc --diag_warning=225 --diag_wrap=off --display_error_number --abi=eabi --preproc_with_compile --preproc_dependency="include/mpu_9150.d_raw" --obj_directory="include" $(GEN_OPTS__FLAG) "$<"
	@echo 'Finished building: "$<"'
	@echo ' '

include/uartstdio.obj: ../include/uartstdio.c $(GEN_OPTS) | $(GEN_HDRS)
	@echo 'Building file: "$<"'
	@echo 'Invoking: ARM Compiler'
	"/Applications/ti/ccsv8/tools/compiler/ti-cgt-arm_18.1.1.LTS/bin/armcl" -mv7M4 --code_state=16 --float_support=FPv4SPD16 -me --include_path="/Users/crohan009/Documents/Stuff/USR18/code base/Data_extraction/i2c" --include_path="/Applications/ti/TivaWare_C_Series-2.1.4.178" --include_path="/Applications/ti/TivaWare_C_Series-2.1.4.178/utils" --include_path="/Users/crohan009/Documents/Stuff/USR18/code base/Data_extraction/i2c/include" --include_path="/Applications/ti/ccsv8/tools/compiler/ti-cgt-arm_18.1.1.LTS/include" --define=ccs="ccs" --define=PART_TM4C123GE6PM -g --gcc --diag_warning=225 --diag_wrap=off --display_error_number --abi=eabi --preproc_with_compile --preproc_dependency="include/uartstdio.d_raw" --obj_directory="include" $(GEN_OPTS__FLAG) "$<"
	@echo 'Finished building: "$<"'
	@echo ' '


