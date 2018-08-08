#ifndef I2C_FUNCTIONS_H
#define I2C_FUNCTIONS_H

#ifdef __cplusplus
extern "C" {
#endif

void delay_ms(unsigned long num_ms);
long millis();

int i2c_write(unsigned char slave_addr, unsigned char reg_addr,
		unsigned char length, unsigned char const *data);
int i2c_read(unsigned char slave_addr, unsigned char reg_addr,
		unsigned char length, unsigned char *data);
int i2c_write_lsm(unsigned char slave_addr, unsigned char reg_addr,
        unsigned char length, unsigned char const *data);
int i2c_read_lsm(unsigned char slave_addr, unsigned char reg_addr,
        unsigned char length, unsigned char *data);

#define __SAM3X8E__

#ifdef __cplusplus
};
#endif


#endif
