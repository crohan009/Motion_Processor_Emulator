#ifndef SYSTICK_H
#define SYSTICK_H

// Delay for a specified number of milliseconds
void delay_ms(unsigned long num_ms);

// Returns the current time in milliseconds
long get_ms();

// Pauses until the time (in milliseconds) given in the argument
void wait_until(long t);

#endif
