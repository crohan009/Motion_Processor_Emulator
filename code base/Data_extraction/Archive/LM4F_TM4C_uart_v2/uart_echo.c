//*****************************************************************************
//
// uart_echo.c - Example for reading data from and writing data to the UART in
//               an interrupt driven fashion.
//
// Copyright (c) 2011-2012 Texas Instruments Incorporated.  All rights reserved.
// Software License Agreement
//
// Texas Instruments (TI) is supplying this software for use solely and
// exclusively on TI's microcontroller products. The software is owned by
// TI and/or its suppliers, and is protected under applicable copyright
// laws. You may not combine this software with "viral" open-source
// software in order to form a larger program.
//
// THIS SOFTWARE IS PROVIDED "AS IS" AND WITH ALL FAULTS.
// NO WARRANTIES, WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING, BUT
// NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE APPLY TO THIS SOFTWARE. TI SHALL NOT, UNDER ANY
// CIRCUMSTANCES, BE LIABLE FOR SPECIAL, INCIDENTAL, OR CONSEQUENTIAL
// DAMAGES, FOR ANY REASON WHATSOEVER.
//
// This is part of revision 9453 of the EK-LM4F232 Firmware Package.
//
//*****************************************************************************
# define PI           3.14159265358979323846  /* pi */

#include <stdio.h>
#include <errno.h>
#include <stdbool.h>
#include <stdarg.h>
#include <stdint.h>
#include <math.h>
#include "inc/hw_ints.h"
#include "inc/hw_memmap.h"
#include "inc/hw_types.h"
#include "inc/hw_i2c.h"
#include "driverlib/debug.h"
#include "driverlib/fpu.h"
#include "driverlib/gpio.h"
#include "driverlib/interrupt.h"
#include "driverlib/sysctl.h"
#include "driverlib/uart.h"
#include "driverlib/rom.h"
#include "driverlib/systick.h"
#include "driverlib/pin_map.h"
#include "driverlib/uart.h"
#include "driverlib/i2c.h"
#include "include/uartstdio.h"
//#include "grlib/grlib.h"
//#include "drivers/cfal96x64x16.h"


//#include "LM4F_TM4C_libs.h"
//#include "systick.h"
#include "MPU_interface/mpu_interface.h"
#include "i2c_functions/i2c_TM4C.h"
#include "i2c_functions/i2c_functions.h"
#include "include/imu.h"


int *f;
uint8_t *data = 0;
imu_scaled_data b;
int result;
long gyro[3], accel[3];
float gyro_offsets[3] = {0, 0, 0};
uint32_t i;
float lsm_gx, lsm_gy, lsm_gz, lsm_mx, lsm_my, lsm_mz;


// Magnetometer hard iron calibration offsets :
//offset_x = -22.45
//offset_y = 21.42
//offset_z = -28.13

// Magnetometer soft iron calibration offsets :
//scale_x = 1.00
//scale_y = 0.99
//scale_z = 1.02

float lsm_mag_sensitivity_scale_factor[3] = {1.00, 0.99, 1.02};
float lsm_magbias[3] = {-22.45, 21.42, -28.13};

typedef unsigned char uint8;

void print_float(double v)
{
  int decimal = 2;
  int i = 1;
  int intPart, fractPart;
  for (;decimal!=0; i*=10, decimal--);
  intPart = (int)v;
  fractPart = (int)((v-(double)(int)v)*i);
  if(fractPart < 0) fractPart *= -1;
  UARTprintf("%i.%i", intPart, fractPart);
}

void ConfigureUART(void)
{
    //UART0
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOA);// Enable the GPIO Peripheral used by the UART.
    SysCtlPeripheralEnable(SYSCTL_PERIPH_UART0); // Enable UART0

    // Configure GPIO Pins for UART mode.
    GPIOPinConfigure(GPIO_PA0_U0RX);
    GPIOPinConfigure(GPIO_PA1_U0TX);
    GPIOPinTypeUART(GPIO_PORTA_BASE, GPIO_PIN_0 | GPIO_PIN_1);

    UARTClockSourceSet(UART0_BASE, UART_CLOCK_PIOSC);    // Use the internal 16MHz oscillator as the UART clock source.
    UARTStdioConfig(0, 115200, 16000000);    // Initialize the UART for console I/O.
}


//typedef unsigned char uint8;

#define USE_PLL_CLOCK

#define AP_UART UART0_BASE

// send 1 byte to AP_UART
void AP_send_char(char s)
{
    ROM_UARTCharPut(AP_UART,s);
}

// send n bytes to AP_UART
void AP_send(int n, const char *s)
{
    int i;
    for (i=0; i<n; i++)
    {
        ROM_UARTCharPut(AP_UART,s[i]);
    }
}

// send a string message to AP_UART
void AP_send_msg(const char *s)
{
  AP_send(strlen(s), s);
}

// send a string message to UART0
void send_msg(const char *s)
{
  send(strlen(s), s);
}

void send_char(char s)
{
    ROM_UARTCharPut(UART0_BASE,s);
}

// send n bytes to AP_UART
void send(int n, const char *s)
{
    int i;
    for (i=0; i<n; i++)
    {
        ROM_UARTCharPut(UART0_BASE,s[i]);
    }
}


// some simple delay functions
double delay()
{
    double _s = 0;
    int k;
    for (k=0; k<50000; k++) _s += 1./k;
    return _s;
}
double delay2(int k1)
{
    double _s = 0;
    int k;
    for (k=0; k<500*k1; k++) _s += 1./k;
    return _s;
}


// system tick (systick)

static volatile long systick_n = 0;

void systick_int_handler()
{
    systick_n ++;
}

void enable_systick()
{
//    AP_send_msg("enabling systick");
    SysTickPeriodSet(160000);
    SysTickIntRegister(&systick_int_handler);
    SysTickIntEnable();
    SysTickEnable();
}

// Delay for a specified number of milliseconds
void delay_ms(unsigned long num_ms){
    int systick_n1 = systick_n;
    while (systick_n < (systick_n1 + num_ms));
}

// Returns the current time in milliseconds
long get_ms()
{
    return systick_n;
}

// Pauses until the time (in milliseconds) given in the argument
void wait_until(long t)
{
    while (systick_n < t);
}


void calibrate_lsm_gyro(float gyro_offsets[3]){        // Calibrates the LSM gyrometer data with 500 samples
    for(i = 0; i<2000; i++){
    		//long _t = get_ms();
        read_scaled_imu_data(&b);
        gyro_offsets[0] += b.gyro_data[0];
        gyro_offsets[1] += b.gyro_data[1];
        gyro_offsets[2] += b.gyro_data[2];
        //wait_until(_t + 5);   // wait for 5 milliseconds
    }
    gyro_offsets[0] /= 2000;
    gyro_offsets[1] /= 2000;
    gyro_offsets[2] /= 2000;

    char gyr_off[50];
    sprintf(gyr_off, "lsm gyrometer offsets: \t%.2f\t%.2f\t%.2f\r\n", gyro_offsets[0], gyro_offsets[1], gyro_offsets[2]);
    send_msg(gyr_off);
}

void test_invensense()
{


	send_msg("######################## MPU-9150 ########################\r\n");
    unsigned char _whoami;
    unsigned char _whoami_lsm;
    i2c_common_setup();
    char who[64];
    i2c_read(104,117,1,&_whoami);
    sprintf(who,"Device ID : 0x%02x \r\n",_whoami);
    send_msg(who);
    mpu_open(200,1);
    send_msg("MPU initialized \r\n");




    send_msg("######################## LSM9DS0 ########################\r\n");
    char who_lsm[64];
    i2c_read(ADDR_ACCEL_LSM9,0x0f,1,&_whoami_lsm);
    sprintf(who_lsm,"Device ID : 0x%02x \r\n",_whoami_lsm);
    send_msg(who_lsm);
    delay_ms(500);
    init_imu_scaledStruct(&b, SCALE_2G, SCALE_2GAUSS, SCALE_250_DPS, UNIT_M_SQUARED);
    init_imu(&b);
    delay_ms(300);
    calibrate_lsm_gyro(&gyro_offsets);
    delay_ms(300);
    send_msg("LSM initialized \r\n");

    EulerAngles_outputs angs;
    MPU_MPL_outputs mpu_mpl_outputs;
    RotationMatrix_outputs R;

    send_msg("reading IMU data... \r\n");
    while (1)
    {
        long _t = get_ms();
        read_scaled_imu_data(&b);
        int data_updated = mpu_update(&mpu_mpl_outputs);
//        mpu_get_euler(&angs); // angs.angles[3] now has Euler angles (in radians)

        char s[200];                // char buffer


        lsm_gx = (b.gyro_data[0] - gyro_offsets[0]) * PI/180;
		lsm_gy = (b.gyro_data[1] - gyro_offsets[1]) * PI/180;
		lsm_gz = (b.gyro_data[2] - gyro_offsets[2]) * PI/180;

		lsm_mx = (b.mag_data[0]*100 - lsm_magbias[0]) * lsm_mag_sensitivity_scale_factor[0];                     // mag x, y, z in micro-Tesla
		lsm_my = (b.mag_data[1]*100 - lsm_magbias[1]) * lsm_mag_sensitivity_scale_factor[1];
		lsm_mz = (b.mag_data[2]*100 - lsm_magbias[2]) * lsm_mag_sensitivity_scale_factor[2];

        // LSM_raw: \t
        sprintf(s,"LSM_raw: \t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\r\n",
                                                    b.acc1_data[0], b.acc1_data[1], b.acc1_data[2],
												   lsm_gx, lsm_gy, lsm_gz,
												   lsm_mx, lsm_my, lsm_mz);
		send_msg(s);


//        // LSM magnetometer values (for the purpose of calibration) in micro-Tesla
//		sprintf(s,"%.2f\t%.2f\t%.2f\r\n",
//				b.mag_data[0]*100, b.mag_data[1]*100, b.mag_data[2]*100);
//		send_msg(s);


		//MPU_raw: \t
		sprintf(s,"MPU_raw: \t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\r\n",
												mpu_mpl_outputs.acc[0], mpu_mpl_outputs.acc[1], mpu_mpl_outputs.acc[2],
												mpu_mpl_outputs.gyr[0], mpu_mpl_outputs.gyr[1], mpu_mpl_outputs.gyr[2],
												mpu_mpl_outputs.mag[0], mpu_mpl_outputs.mag[1], mpu_mpl_outputs.mag[2]);
        send_msg(s);

//        MPU_pry: \t
//        sprintf(/s,"MPU_ypr: \t%.2f\t%.2f\t%.2f\r\n",
//        											angs.angles[0]*180.0/3.1415926, angs.angles[1]*180.0/3.1415926, angs.angles[2]*180.0/3.1415926);
//        send_msg(s);

        mpu_get_rot_mat(&R);  // R.R[9] now has rotation matrix elements
        wait_until(_t + 5);   // wait for 5 milliseconds
    }
}


//*****************************************************************************
//
// The UART interrupt handlers.
//
//*****************************************************************************
void
UART0IntHandler(void)
{
    unsigned long ulStatus; // Get the interrupt status.
    ulStatus = ROM_UARTIntStatus(UART0_BASE, true); // Clear the asserted interrupts.
    ROM_UARTIntClear(UART0_BASE, ulStatus);  // Loop while there are characters in the receive FIFO.
    while(ROM_UARTCharsAvail(UART0_BASE))
    {
        unsigned char c = ROM_UARTCharGetNonBlocking(UART0_BASE); // Read the next character from the UART and write it back to the UART
        //ROM_UARTCharPutNonBlocking(UART2_BASE, c);
    }
}




int main(void)
{
  // initialize communications structures
      //init_AP_comms();

    //
    // Enable lazy stacking for interrupt handlers.  This allows floating-point
    // instructions to be used within interrupt handlers, but at the expense of
    // extra stack usage.
    //
    ROM_FPULazyStackingEnable();

#ifdef USE_PLL_CLOCK
    ROM_SysCtlClockSet(SYSCTL_SYSDIV_2_5 | SYSCTL_USE_PLL | SYSCTL_OSC_MAIN | SYSCTL_XTAL_16MHZ);
#else
    ROM_SysCtlClockSet(SYSCTL_SYSDIV_1 | SYSCTL_USE_OSC | SYSCTL_OSC_MAIN | SYSCTL_XTAL_16MHZ);
#endif

    ConfigureUART();
    enable_systick();


    ////
    //
//    test_lsm();
    test_invensense();
   while (1)
   {
     delay();
//     AP_send_msg("hello");
   }
   // to send positioning_update message:
   // positioning_update_t positioning_update_msg;
   // AP_send_positioning_update_msg(&positioning_update_msg);
}
