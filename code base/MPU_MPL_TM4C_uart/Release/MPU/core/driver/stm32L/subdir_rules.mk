################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Each subdirectory must supply rules for building sources it contributes
MPU/core/driver/stm32L/log_stm32.obj: ../MPU/core/driver/stm32L/log_stm32.c $(GEN_OPTS) | $(GEN_HDRS)
	@echo 'Building file: "$<"'
	@echo 'Invoking: ARM Compiler'
	"/Applications/ti/ccsv8/tools/compiler/ti-cgt-arm_18.1.1.LTS/bin/armcl" -mv7M4 --code_state=16 --float_support=FPv4SPD16 --abi=eabi -me -O2 --include_path="/Applications/ti/ccsv8/tools/compiler/ti-cgt-arm_18.1.1.LTS/include" --include_path="C:/StellarisWare/boards/ek-lm4f232" --include_path="C:/StellarisWare/inc" --include_path="C:/StellarisWare" --gcc --define=ccs="ccs" --define=PART_LM4F232H5QD --define=TARGET_IS_BLIZZARD_RA1 --diag_warning=225 --display_error_number --gen_func_subsections=on --ual --preproc_with_compile --preproc_dependency="MPU/core/driver/stm32L/log_stm32.d_raw" --obj_directory="MPU/core/driver/stm32L" $(GEN_OPTS__FLAG) "$<"
	@echo 'Finished building: "$<"'
	@echo ' '


