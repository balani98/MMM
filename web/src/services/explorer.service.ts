import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { ExplorerInput } from 'src/app/models/explorerInput';

@Injectable({
  providedIn: 'root'
})
export class ExplorerService {
  httpOptions = {
    headers: new HttpHeaders({
      'Content-Type': 'application/json',
      "Access-Control-Allow-Origin":'*' // Example: Setting the Content-Type header
    })
  };
  constructor(private http:HttpClient) { }
  //uri = 'https://global-mso-api-lr4grty6xq-uc.a.run.app';
  uri = 'http://34.69.111.69:5000';
  //uri = 'http://localhost:8000'
  dateCheck(dateSelector:any){
    var body = {
      'date_selector':dateSelector
    }
    return this.http.post(`${this.uri}/api/explorer/datecheck`,body,{
      observe:'events'
    })
  }

  investmentCheck(investmentSelector:any){
    var body = {
      'investment_selector':investmentSelector
    }
    return this.http.post(`${this.uri}/api/explorer/investmentcheck`,body,this.httpOptions);
  }

  targetCheck(targetSelector:any){
    var body = {
      'target_selector':targetSelector
    }
    return this.http.post(`${this.uri}/api/explorer/targetcheck`,body,this.httpOptions);
  }
  generateEDAreport(explorerInputs:ExplorerInput){
    var body = explorerInputs
    return this.http.post(`${this.uri}/api/explorer/generateEDAReport`,body,this.httpOptions);
  }
  downloadEDAreport(){
    var headers = {
      "Access-Control-Allow-Origin":'*' // Example: Setting the Content-Type header
    }
    return this.http.get(`${this.uri}/api/explorer/downloadEDAReport`,{ headers, responseType: 'text' });
  }

  downloadUserguide(){
    var headers = {
      "Access-Control-Allow-Origin":'*' // Example: Setting the Content-Type header
    }
    return this.http.get(`${this.uri}/api/explorer/downloadUserguide`,{ headers, responseType: 'blob' });
  }
}
