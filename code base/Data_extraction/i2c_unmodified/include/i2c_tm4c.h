#include <hal_common_includes.h>
#include <stdint.h>

void InitI2C0(void);
int i2c_write_reg(uint8_t addr, uint8_t reg, uint8_t val, int *flag);
uint32_t I2CReceive(uint32_t slave_addr, uint8_t reg);