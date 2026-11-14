"""
LED wiring test (run on the Pi).

Wiring:  GPIO17 (physical pin 11) -> 220 ohm resistor -> LED long leg (+)
         LED short leg (-) -> GND (physical pin 9)
Usage:   python pi/led_test.py            (blinks 5 times)
         python pi/led_test.py --pin 27   (other GPIO pin)
"""
import argparse
import time

from gpiozero import LED

ap = argparse.ArgumentParser()
ap.add_argument("--pin", type=int, default=17)
args = ap.parse_args()

led = LED(args.pin)
print(f"Blinking LED on GPIO{args.pin} ...")
for i in range(5):
    led.on()
    time.sleep(0.5)
    led.off()
    time.sleep(0.5)
print("If the LED did not light: check the long/short leg direction and the resistor.")
