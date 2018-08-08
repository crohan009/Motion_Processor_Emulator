#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include "inc/hw_types.h"
#include "inc/hw_memmap.h"
#include "inc/hw_i2c.h"
#include "driverlib/sysctl.h"
#include "driverlib/i2c.h"
#include "driverlib/gpio.h"
#include "driverlib/rom.h"
#include "driverlib/pin_map.h"
#include "i2c_TM4C.h"


//#define I2C_MASTER_BASE_ADDR I2C0_MASTER_BASE
#define I2C_MASTER_BASE_ADDR I2C0_BASE

void AP_send_msg(const char *s);

void i2c_common_setup(void)
{
	int i;
	//for (i=0; i<2; i++) SysCtlDelay(300000L);
    SysCtlPeripheralEnable(SYSCTL_PERIPH_I2C0);        //Enable clock to I2C
    SysCtlPeripheralReset(SYSCTL_PERIPH_I2C0);
	SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOB);       //clock for the pins
	GPIOPinConfigure(GPIO_PB2_I2C0SCL);
	GPIOPinConfigure(GPIO_PB3_I2C0SDA);
	GPIOPinTypeI2CSCL(GPIO_PORTB_BASE,GPIO_PIN_2);     //GPIO to I2C
	GPIOPinTypeI2C(GPIO_PORTB_BASE, GPIO_PIN_3);     //GPIO to I2C
    I2CMasterInitExpClk(I2C_MASTER_BASE_ADDR, SysCtlClockGet(), true); // fast I2C mode (400 kHz) if third argument is true
    HWREG(I2C_MASTER_BASE_ADDR + I2C_O_FIFOCTL) = 80008000;

	//GPIODirModeSet(GPIO_PORTB_BASE,GPIO_PIN_2,GPIO_DIR_MODE_HW);
	//GPIODirModeSet(GPIO_PORTB_BASE,GPIO_PIN_3,GPIO_DIR_MODE_HW);
	//I2CMasterEnable(I2C_MASTER_BASE_ADDR);          //Enable Master

	SysCtlDelay(32000L);
}



int i2c_write_reg(uint32 addr, uint8_t reg, uint8_t val, int *flag)
{
	I2CMasterSlaveAddrSet(I2C_MASTER_BASE_ADDR, addr, false);
	while(I2CMasterBusy(I2C_MASTER_BASE_ADDR)) { }
	I2CMasterDataPut(I2C_MASTER_BASE_ADDR, reg);
	I2CMasterControl(I2C_MASTER_BASE_ADDR, I2C_MASTER_CMD_BURST_SEND_START);
	while(I2CMasterBusy(I2C_MASTER_BASE_ADDR)) { }
	//SysCtlDelay(300L);
	I2CMasterDataPut(I2C_MASTER_BASE_ADDR, val);
	//I2CMasterControl(I2C_MASTER_BASE_ADDR, I2C_MASTER_CMD_BURST_SEND_CONT);
	while(I2CMasterBusy(I2C_MASTER_BASE_ADDR)) { }
	I2CMasterControl(I2C_MASTER_BASE_ADDR, I2C_MASTER_CMD_BURST_SEND_FINISH);
	return 1;
}


int i2c_write_bytes(uint32 addr, uint8_t reg, int n,  uint8_t const *val, int *flag)
{
	I2CMasterSlaveAddrSet(I2C_MASTER_BASE_ADDR, addr, false);
	while(I2CMasterBusy(I2C_MASTER_BASE_ADDR)) { }
	I2CMasterDataPut(I2C_MASTER_BASE_ADDR, reg);
	I2CMasterControl(I2C_MASTER_BASE_ADDR, I2C_MASTER_CMD_BURST_SEND_START);
	while(I2CMasterBusy(I2C_MASTER_BASE_ADDR)) { }
	int i;
	for (i=0; i<(n-1); i++)
	{
		//SysCtlDelay(25L);
		I2CMasterDataPut(I2C_MASTER_BASE_ADDR, val[i]);
		I2CMasterControl(I2C_MASTER_BASE_ADDR, I2C_MASTER_CMD_BURST_SEND_CONT);
		while(I2CMasterBusy(I2C_MASTER_BASE_ADDR)) { }
	}
	I2CMasterDataPut(I2C_MASTER_BASE_ADDR, val[n-1]);
	I2CMasterControl(I2C_MASTER_BASE_ADDR, I2C_MASTER_CMD_BURST_SEND_FINISH);
	while(I2CMasterBusy(I2C_MASTER_BASE_ADDR)) { }
	//SysCtlDelay(300L);
	return 1;
}


int i2c_read_bytes(uint32 addr, uint8_t reg, int n, uint8_t *ret, int *flag)
{
	I2CMasterSlaveAddrSet(I2C_MASTER_BASE_ADDR, addr, false);
	while(I2CMasterBusy(I2C_MASTER_BASE_ADDR)) { }
	I2CMasterDataPut(I2C_MASTER_BASE_ADDR, reg);
	I2CMasterControl(I2C_MASTER_BASE_ADDR, I2C_MASTER_CMD_SINGLE_SEND);
	while(I2CMasterBusy(I2C_MASTER_BASE_ADDR)) { }
	I2CMasterSlaveAddrSet(I2C_MASTER_BASE_ADDR, addr, true);
	while(I2CMasterBusy(I2C_MASTER_BASE_ADDR)) { }
	I2CMasterControl(I2C_MASTER_BASE_ADDR, I2C_MASTER_CMD_BURST_RECEIVE_START);
	int i;
	for (i=0; i<(n-1); i++)
	{
		while(I2CMasterBusy(I2C_MASTER_BASE_ADDR)) { }
		ret[i] = I2CMasterDataGet(I2C_MASTER_BASE_ADDR);
		if (i<(n-2)) I2CMasterControl(I2C_MASTER_BASE_ADDR, I2C_MASTER_CMD_BURST_RECEIVE_CONT);
	}
	I2CMasterControl(I2C_MASTER_BASE_ADDR, I2C_MASTER_CMD_BURST_RECEIVE_FINISH);
	while(I2CMasterBusy(I2C_MASTER_BASE_ADDR)) { }
	ret[n-1] = I2CMasterDataGet(I2C_MASTER_BASE_ADDR);
	SysCtlDelay(3L);
//	 char s[64];
//	sprintf(s,"A %x",ret[0]);
	//AP_send_msg(s);
	return 1;
}


uint32_t I2CReceive(uint32_t slave_addr, uint8_t reg)
{
    I2CMasterSlaveAddrSet(I2C0_BASE, slave_addr, false); //specify that we are writing (a register address) to the slave device
    I2CMasterDataPut(I2C0_BASE, reg); //specify register to be read
    I2CMasterControl(I2C0_BASE, I2C_MASTER_CMD_BURST_SEND_START); //send control byte and register address byte to slave device
    while(I2CMasterBusy(I2C0_BASE)); //wait for MCU to finish transaction
    I2CMasterSlaveAddrSet(I2C0_BASE, slave_addr, true); //specify that we are going to read from slave device
    I2CMasterControl(I2C0_BASE, I2C_MASTER_CMD_SINGLE_RECEIVE); //send control byte and read from the register we specified
    while(I2CMasterBusy(I2C0_BASE));     //wait for MCU to finish transaction
    return I2CMasterDataGet(I2C0_BASE); //return data pulled from the specified register
}
