import { Component, EventEmitter, OnInit, Output } from '@angular/core';
import { FormControl } from '@angular/forms';
import { getData } from 'src/data/data';
import { PredictorService } from 'src/services/predictor.service';
import { WebSocketService } from 'src/services/web-socket.service';
import { Alert } from './alert.model';

@Component({
  selector: 'app-predictor-form',
  templateUrl: './predictor-form.component.html',
  styleUrls: ['./predictor-form.component.scss'],
})
export class PredictorFormComponent implements OnInit {
  constructor(private predictorService: PredictorService,private websocketService:WebSocketService) {} 
  @Output() responseCurvesData = new EventEmitter<any>();
  @Output() effectiveSharesData = new EventEmitter<any>();
  notifications: Alert[] = [];
  countries = new FormControl('');
  contextual_vars = new FormControl('');
  adstocklist_vars = new FormControl('');
  controlVariables: string[] = [];
  adstockList: string[] = [];
  checkTheCheckBox: boolean = true;
  countriesList: string[] = [
    'ARGENTINA',
    'ARUBA',
    'AUSTRALIA',
    'AUSTRIA',
    'BANGLADESH',
    'BELGIUM',
    'BRAZIL',
    'CANADA',
    'CHILE',
    'CHINA',
    'COLOMBIA',
    'CROATIA',
    'CZECHIA',
    'DENMARK',
    'DOMINICAN REPUBLIC',
    'EGYPT',
    'ESTONIA',
    'FINLAND',
    'GERMANY',
    'GREECE',
    'HONG KONG',
    'HUNGARY',
    'ICELAND',
    'INDIA',
    'INDONESIA',
    'IRELAND',
    'ISRAEL',
    'ITALY',
    'KENYA',
    'LITHUANIA',
    'LUXEMBOURG',
    'MALAYSIA',
    'MEXICO',
    'NETHERLANDS',
    'NEW ZEALAND',
    'NICARAGUA',
    'NIGERIA',
    'NORWAY',
    'PAKISTAN',
    'PARAGUAY',
    'PERU',
    'PHILIPPINES',
    'POLAND',
    'PORTUGAL',
    'RUSSIA',
    'SINGAPORE',
    'SLOVAKIA',
    'SLOVENIA',
    'SOUTH AFRICA',
    'SOUTH KOREA',
    'SPAIN',
    'SWEDEN',
    'SWITZERLAND',
    'THAILAND',
    'UNITED KINGDOM',
    'UNITED STATES OF AMERICA',
    'VIETNAM',
  ];

  getPredictorUserInputs() {
    this.predictorService.getPredictorInputs().subscribe(
      (res: any) => {
        if (res.status === 200) {
          console.log('success');
          this.countriesList = res.body.user_options.country;
          this.controlVariables = res.body.user_options.control_variables;
          this.adstockList = res.body.user_options.adstock;
        }
      },
      (errorResponse: any) => {
        console.log('failure');
      }
    );
  }
  onSubmit(predictorInputsValue: any) {
    console.log(predictorInputsValue)
    var predictor_user_input = {
      'include_holiday': predictorInputsValue.holiday_selector,
      'country': predictorInputsValue.country_selector,
      'control_variables':  predictorInputsValue.contextual_vars_selector,
      'adstock': predictorInputsValue.adstock_selector
      }
    this.predictorService.submitPredictorInputs(predictor_user_input).subscribe(
      (res: any) => {
        setTimeout(()=>{
          alert('Model is building in background !! you can check the result in Show Results')
      },1000)

      },
      (errorResponse: any) => {
        console.log('failure');
      }
    );
  }
  ngOnInit(): void {
    this.getPredictorUserInputs();
    this.websocketService.listen('notification').subscribe((data) => {
      console.log('deeps',data)
      this.notifications.push(data);
      alert(data.message)
      
    });
  }
}
