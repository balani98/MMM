import { HttpResponse } from '@angular/common/http';
import { Component, OnInit, forwardRef } from '@angular/core';
import { ExplorerService } from 'src/services/explorer.service';
import { FileUploadService } from 'src/services/file-upload.service';
import { ExplorerInput } from '../models/explorerInput';
import { Output, EventEmitter } from '@angular/core';
import { NgxSpinnerService } from 'ngx-spinner';
import {
  faFileUpload,faFileArrowUp,faFileCsv
} from '@fortawesome/free-solid-svg-icons';
@Component({
  selector: 'app-explorer-form',
  templateUrl: './explorer-form.component.html',
  styleUrls: ['./explorer-form.component.scss'],
})
export class ExplorerFormComponent implements OnInit {
  faFileUpload = faFileCsv
  protected explorerInput = new ExplorerInput();
  @Output() overviewStats = new EventEmitter<string>();
  @Output() variableStats = new EventEmitter<string>();
  @Output() variableName = new EventEmitter<string>();
  @Output() histogram = new EventEmitter<string>();
  @Output() UIStats = new EventEmitter<string>();
  @Output() currencyTypeEmitter = new EventEmitter<string>();
  @Output() validationReport = new EventEmitter<string>();
  @Output() validationReportZeroDataPointsEmitter = new EventEmitter<string>();
  @Output() validationReportMissingDataPointsEmitter = new EventEmitter<string>();
  @Output() validationReportOutliersEmitter = new EventEmitter<string>();
  @Output() validationReportVarianceVariablesEmitter = new EventEmitter<string>();
  @Output() edaGeneration = new EventEmitter<boolean>();
  file: File;
  dataGranularityValue:string =""
  name_of_file: any = 'No file chosen...';
  selectorsList: any = [];
  currencyType:any = 'USD';
  dateSelectorsList:any = [];
  spendSelectorsList:any = [];
  targetSelectorsList:any = [];
  dateSelectorError: string = '';
  dateSelectorErrorBool: boolean = false;
  dateSelected: boolean = false;
  investmentSelectorError: string = '';
  investmentSelectorErrorBool: boolean = false;
  targetSelectorError: string = '';
  targetSelectorErrorBool: boolean = false;
  overview_stats: any;
  variable_stats: any;
  histogramStats: any;
  defaultCurrency:string='USD';
  validation_report:any = {}
  validationReportZeroDataPoints:any={}
  validationReportMissingDataPoints:any={}
  validationReportOutliers:any={}
  validationReportVarianceVariables:any={}
  constructor(
    private fileUploadService: FileUploadService,
    private explorerService: ExplorerService,
    private spinner: NgxSpinnerService
  ) {}
  checkFile(event: any) {
    this.file = event.target.files[0];
    this.name_of_file = this.file.name;
    this.uploadFile()
  }
  uploadFile() {
    this.spinner.show()
    const fd = new FormData();
    fd.append('csv', this.file, this.file.name);
    this.fileUploadService.uploadFile(fd).subscribe((response: any) => {
      console.log(response)
      if (response.message === "uploaded successfully") {
        this.spinner.hide()
        alert('file uploaded succesfully');
        this.selectorsList = response.columns;
        this.dateSelectorsList = response.user_options.date;
        this.spendSelectorsList = response.user_options.channel_spend.sort();
        this.targetSelectorsList = response.user_options.target.sort();

      }
    });
  }
  fileIsUploaded() {
    let result = false;
    if (this.file && this.file != null) {
      result = true;
    }
    return result;
  }
  dateCheck(event: any) {
    var dateSelector = event.target.value;
    this.explorerService.dateCheck(dateSelector).subscribe(
      (res: any) => {
        if (res.status === 200) {
          this.dateSelectorError = '';
          this.dateSelectorErrorBool = false;
          this.dateSelected = true;
          this.dataGranularityValue = res.body.data_granularity
        }
      },
      (errorResponse: any) => {
        this.dateSelected = false;
        this.dateSelectorError = errorResponse.error.error;
        this.dateSelectorErrorBool = true;
      }
    );
  }
  investmentCheck(event: any) {
    console.log(this.investmentCheck);
    var investmentSelector = event.value;
    this.explorerService.investmentCheck(investmentSelector).subscribe(
      (res: any) => {
        if (res.status === 200) {
          this.investmentSelectorError = '';
          this.investmentSelectorErrorBool = false;
        }
      },
      (errorResponse: any) => {
        this.investmentSelectorError = errorResponse.error.error;
        this.investmentSelectorErrorBool = true;
      }
    );
  }
  targetCheck(event: any) {
    var targetSelector = event.target.value;
    console.log(targetSelector);
    this.explorerService.targetCheck(targetSelector).subscribe(
      (res: any) => {
        if (res.status === 200) {
          this.targetSelectorError = '';
          this.targetSelectorErrorBool = false;
        }
      },
      (errorResponse: any) => {
        this.targetSelectorError = errorResponse.error.error;
        this.targetSelectorErrorBool = true;
      }
    );
  }
  showSpinner() {
    this.spinner.show();
    setTimeout(() => {
        /** spinner ends after 5 seconds */
        this.spinner.hide();
    }, 5000);
  }
  onSubmit(explorerInputForm: any) {
    this.spinner.show()
    var explorerInputs: ExplorerInput = new ExplorerInput();
    // to get the value of disabled option 
    explorerInputs.dataGranularity =
      explorerInputForm.form.get('data_granularity_selector').value;
    // explorerInputs.currencySelector =
    // explorerInputForm.form.get('currency_selector').value;
    this.currencyType =   explorerInputForm.form.get('currency_selector').value;
    explorerInputs.dateSelector = explorerInputForm.value.date_selector;
    explorerInputs.investmentSelector =
      explorerInputForm.value.investment_selector;
    explorerInputs.targetSelector = explorerInputForm.value.target_selector;
    explorerInputs.targetTypeSelector =
      explorerInputForm.value.target_type_selector;
    this.explorerService.generateEDAreport(explorerInputs).subscribe(
      (res: any) => {
        if (res.status === 200) {
          this.spinner.hide()
          console.log('success');
          alert("EDA Generated");
          console.log(res.sample_report);
          this.validation_report = res.validation_report
          this.overview_stats = res.sample_report['overview'];
          this.variable_stats = res.sample_report['variable'];
          this.histogramStats = res.sample_report['variable'].histogram;
          this.validationReportZeroDataPoints = this.validation_report?.zero_datapoints
          this.validationReportMissingDataPoints = this.validation_report?.missing_datapoints
          this.validationReportOutliers = this.validation_report?.outliers
          this.validationReportVarianceVariables = this.validation_report?.no_variance_var;
          console.log(this.variable_stats)
          this.overviewStats.emit(this.overview_stats);
          this.variableStats.emit(this.variable_stats);
          this.variableName.emit(this.variable_stats.variable_name);
          this.histogram.emit(this.histogramStats);
          this.UIStats.emit(res.UI_stats);
          this.validationReport.emit(this.validation_report)
          this.validationReportZeroDataPointsEmitter.emit(this.validationReportZeroDataPoints)
          this.validationReportMissingDataPointsEmitter.emit(this.validationReportMissingDataPoints)
          this.validationReportOutliersEmitter.emit(this.validationReportOutliers)
          this.validationReportVarianceVariablesEmitter.emit(this.validationReportVarianceVariables)
          this.currencyTypeEmitter.emit(this.currencyType)
          this.edaGeneration.emit(true)
        }
      },
      (errorResponse: any) => {
        console.log('failure');
      }
    );
  }

  ngOnInit(): void {}
}
