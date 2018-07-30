#include "i2c_tm4c.h" 

void InitI2C0(void)
{  
    SysCtlPeripheralEnable(SYSCTL_PERIPH_I2C0); //enable I2C module 0 
    SysCtlPeripheralReset(SYSCTL_PERIPH_I2C0); //reset module 
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOB); //enable GPIO peripheral that contains I2C 0
    GPIOPinConfigure(GPIO_PB2_I2C0SCL); // Configure the pin muxing for I2C0 functions on port B2 and B3.
    GPIOPinConfigure(GPIO_PB3_I2C0SDA);
    GPIOPinTypeI2CSCL(GPIO_PORTB_BASE, GPIO_PIN_2); // Select the I2C function for these pins.
    GPIOPinTypeI2C(GPIO_PORTB_BASE, GPIO_PIN_3);
    // Enable and initialize the I2C0 master module.  Use the system clock for
    // the I2C0 module.  The last parameter sets the I2C data transfer rate.
    // If false the data rate is set to 100kbps and if true the data rate will
    // be set to 400kbps.
    I2CMasterInitExpClk(I2C0_BASE, SysCtlClockGet(), false);
    HWREG(I2C0_BASE + I2C_O_FIFOCTL) = 80008000; //clear I2C FIFOs
}

int i2c_write_reg(uint8_t addr, uint8_t reg, uint8_t val, int *flag)
{
    I2CMasterSlaveAddrSet(I2C0_BASE, addr, false);
    while(I2CMasterBusy(I2C0_BASE)) { }
    I2CMasterDataPut(I2C0_BASE, reg);
    I2CMasterControl(I2C0_BASE, I2C_MASTER_CMD_BURST_SEND_START);
    while(I2CMasterBusy(I2C0_BASE)) { }
    //SysCtlDelay(300L);
    I2CMasterDataPut(I2C0_BASE, val);
    I2CMasterControl(I2C0_BASE, I2C_MASTER_CMD_BURST_SEND_CONT);
    while(I2CMasterBusy(I2C0_BASE)) { }
    I2CMasterControl(I2C0_BASE, I2C_MASTER_CMD_BURST_SEND_FINISH);
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