import { Pipe, PipeTransform } from '@angular/core';

@Pipe({
  name: 'roundingPipe'
})
export class RoundingPipe implements PipeTransform {
  transform(value: number, digits: number): string {
    // Extract the integer part
    const integerPart = Math.floor(value);
    // Calculate the fractional part
    const fractionalPart = value - integerPart;
    // Round the fractional part to the specified number of digits
    const roundedFraction = fractionalPart.toFixed(digits);
    // Return the integer part concatenated with the rounded fractional part
    return integerPart + parseFloat(roundedFraction).toFixed(digits).substring(1);
  }
}
