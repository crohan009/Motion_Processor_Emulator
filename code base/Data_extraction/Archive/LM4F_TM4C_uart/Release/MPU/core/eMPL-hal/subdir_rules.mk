################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Each subdirectory must supply rules for building sources it contributes
MPU/core/eMPL-hal/eMPL_outputs.obj: ../MPU/core/eMPL-hal/eMPL_outputs.c $(GEN_OPTS) | $(GEN_HDRS)
	@echo 'Building file: "$<"'
	@echo 'Invoking: ARM Compiler'
	"/Applications/ti/ccsv8/tools/compiler/ti-cgt-arm_18.1.1.LTS/bin/armcl" -mv7M4 --code_state=16 --float_support=FPv4SPD16 -me -O2 --fp_mode=relaxed --include_path="/Users/crohan009/Documents/Stuff/USR18/code base/Data_extraction/Archive/LM4F_TM4C_uart/include" --include_path="/Users/crohan009/Documents/Stuff/USR18/code base/Data_extraction/Archive/LM4F_TM4C_uart/i2c_functions" --include_path="/Users/crohan009/Documents/Stuff/USR18/code base/Data_extraction/Archive/LM4F_TM4C_uart/MPU/core/driver/stm32L" --include_path="/Users/crohan009/Documents/Stuff/USR18/code base/Data_extraction/Archive/LM4F_TM4C_uart/MPU/core/eMPL-hal" --include_path="/Users/crohan009/Documents/Stuff/USR18/code base/Data_extraction/Archive/LM4F_TM4C_uart/MPU/core/mpl" --include_path="/Users/crohan009/Documents/Stuff/USR18/code base/Data_extraction/Archive/LM4F_TM4C_uart/MPU/core/driver/eMPL" --include_path="/Users/crohan009/Documents/Stuff/USR18/code base/Data_extraction/Archive/LM4F_TM4C_uart/MPU/core/mllite" --include_path="/Users/crohan009/Documents/Stuff/USR18/code base/Data_extraction/Archive/LM4F_TM4C_uart/MPU/core/driver/include" --include_path="/Applications/ti/ccsv8/tools/compiler/ti-cgt-arm_18.1.1.LTS/include" --include_path="/Applications/ti/TivaWare_C_Series-2.1.4.178" --include_path="/Applications/ti/TivaWare_C_Series-2.1.4.178/utils" --define=__FPU_PRESENT=1 --define=ARM_MATH_CM4 --define=USE_DMP --define=MPL_LOG_NDEBUG=1 --define=EMPL --define=MPU9150 --define=EMPL_TARGET_TM4C --define=ccs="ccs" --define=TARGET_IS_TM4C123_RA3 --define=PART_TM4C123GE6PM --gcc --wchar_t=16 --abi=eabi --preproc_with_compile --preproc_dependency="MPU/core/eMPL-hal/eMPL_outputs.d_raw" --obj_directory="MPU/core/eMPL-hal" $(GEN_OPTS__FLAG) "$<"
	@echo 'Finished building: "$<"'
	@echo ' '


