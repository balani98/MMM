import { Component, Input } from '@angular/core';
@Component({
  selector: 'app-main',
  templateUrl: './main.component.html',
  styleUrls: ['./main.component.scss'],
})
export class MainComponent {
  overviewStats: any = {};
  variableStats: any = {};
  variableName: string = '';
  histogramStats: any = {};
  UIStats: any = {};
  currencyType:string;
  validationReport:any = {}
  passoverviewStats(overviewStats: any) {
    this.overviewStats = overviewStats;
  }
  passVariableStats(overviewStats: any) {
    this.variableStats = overviewStats;
  }
  catchVariableName(variableName: any) {
    this.variableName = variableName;
    console.log(variableName);
  }
  passHistogramStats(histogramStats: any) {
    this.histogramStats = histogramStats;
  }
  passUIStats(UIStats: any) {
    this.UIStats = UIStats;
  }

  passCurrencyType(currencyType: any) {
    this.currencyType = currencyType;
  }
  passValidationReport(validationReport:any){
    this.validationReport = validationReport;
    console.log('deeps',this.validationReport)
  }
}
