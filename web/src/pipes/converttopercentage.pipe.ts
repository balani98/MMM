import { DecimalPipe } from '@angular/common';
import { Pipe, PipeTransform } from '@angular/core';

@Pipe({name: 'convertToPercentage'})
export class ConvertToPercentagePipe implements PipeTransform {

constructor(private decimalPipe: DecimalPipe) {}

transform(value: number): number {
  return value*100 as number;
}

}