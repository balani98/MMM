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
  validationReportZeroDataPoints:any={}
  validationReportMissingDataPoints:any={}
  validationReportOutliers:any={}
  validationReportVarianceVariables:any={}
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
  }
  passValidationReportZeroDataPoints(validationReportZeroDataPoints:any){
    this.validationReportZeroDataPoints = validationReportZeroDataPoints;
  }
  passValidationReportMissingDataPoints(validationReportMissingDataPoints:any){
    this.validationReportMissingDataPoints = validationReportMissingDataPoints;
  }
  passValidationReportOutliers(validationReportOutliers:any){
    this.validationReportOutliers = validationReportOutliers;
    console.log(this.validationReportOutliers)
  }
  passValidationReportVarianceVariables(validationReportVarianceVariables:any){
    this.validationReportVarianceVariables = validationReportVarianceVariables;
    console.log(this.validationReportOutliers)
  }
}
