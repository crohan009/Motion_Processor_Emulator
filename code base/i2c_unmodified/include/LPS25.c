#include "math.h"
#include "i2c_tm4c.h"
#include "LPS25.h"
#include <stdint.h>
#include "driverlib/uart.h"

int *f;
int _address ;

void begin(uint8_t address) {
	_address = address;
	i2c_write_reg(_address,0x10, 0xF,f); // resolution: temp=32, pressure=128
//	i2c_write_reg(_address,0x20, 0x8C,f); // reset
	i2c_write_reg(_address,0x20, 0x82,f); // reset
	delay(1000);
	i2c_write_reg(_address,0x20, 0xCC,f); // 25Hz

}

void whoAmI() {
	UARTprintf("Who_am_I Pressure: %04x\n",I2CReceive(_address, 0X0F));
}

uint8_t status(uint8_t status) {
    int count = 1000;
    uint8_t data = 0xff;
    do {
        data = I2CReceive(_address,0x27);
        --count;
        if (count < 0)
            break;
    } while ((data & status) == 0);

    if (count < 0)
        return -1;
    else
        return 0;
}

float cal_LPS25()
{
    int i = 0;
//    float arr[1000]={0};
    float min = readPressure();;
    float max = 0.0;
    for (i=0;i<100;i++){
        if (readPressure() > max)
               {
             max = readPressure();
               }
          if (readPressure() < min)
               {
             min = readPressure();
               }
//          UARTprintf("i:%d",i);
          delay(10);
    }
    UARTprintf("MAX: ");
    print_float(max);
    UARTprintf(" MIN: ");
    print_float(min);
    UARTprintf("\r\n");
}
double readP() {
	return readPressure();
}

float readPressure() {
	i2c_write_reg(_address,0x21, 0x1,f);

	if (status(0x2) < 0)
		return 0;

	uint8_t pressOutH = I2CReceive(_address,0x2A);
	uint8_t pressOutL = I2CReceive(_address,0x29);
	uint8_t pressOutXL = I2CReceive(_address,0x28);

	long val = (((long)pressOutH << 24) | ((long)pressOutL << 16) | ((long)pressOutXL << 8)) >> 8;
//	int i;
//	long val2 = 0;
//    for(i=0;i<20;i++){
//        val2 = val2 + val;
//        delay(100);
//    }
//    val2 = val2/20.0;
//	print_float(val2/4096.0f);
	UARTprintf("Pressure: ");
	print_float(val/4096.0f);
//	UARTprintf("\r\n");
	return val/4096.0f;

}

float avg(){
    int i = 0;
    float val = 0.0;
    for(i=0;i<100;i++){
        val = val + readPressure();
        delay(10);
    }
    UARTprintf("AVG: ");
    print_float(val/100.0);
    UARTprintf("\r\n");
    return (val/100.0);
}

double readT() {
	return readTemperature();
}

float readTemperature() {
	i2c_write_reg(_address,0x21, 0x1,f);
	if (status(0x1) < 0)
		return 0;

	uint8_t tempOutH = I2CReceive(_address,0x2C);
	uint8_t tempOutL = I2CReceive(_address,0x2B);

	int16_t val = tempOutH << 8 | tempOutL & 0xff;
    print_float(42.5f+val/480.0f);
    UARTprintf("\r\n");
	return 42.5f+val/480.0f;
}

float altitude(float pressure){
//    UARTprintf("P : ");
//    print_float(pressure);
//    float alt  = 145366.45*(1. - pow((pressure/1013.25), 0.190284));
//    alt = alt / 30.48;
////    print_float((145366.45*(1. - pow((pressure/1013.25), 0.190284)))/ 3.280839895);
//    print_float(alt+82.);
//    UARTprintf("\r\n");
//    return alt;
//    print_float(set);
    float p = 0;
//    int i = 0;
//    for(i=0;i<20;i++){
        p = p + pressure;
//        delay(100);
//    }
//    p = p/20.0;
    float alt = (1 - pow(p / 1013.25, 0.190263))*44330.8*100.0;
    UARTprintf(" Alt: ");
    print_float(alt);
    UARTprintf("\r\n");
    return alt;
}





