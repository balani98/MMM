import { DecimalPipe } from '@angular/common';
import { Pipe, PipeTransform } from '@angular/core';

@Pipe({name: 'shortNumber'})
export class ShortNumberPipe implements PipeTransform {

constructor(private decimalPipe: DecimalPipe) {}

transform(value: number, digits?: string): string {
  if (!value) {
    return '';
  }

  return this.decimalPipe.transform(value / 1000, digits) as string;
}

}