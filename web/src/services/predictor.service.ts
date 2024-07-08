import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class PredictorService {

  httpOptions = {
    headers: new HttpHeaders({
      'Content-Type': 'application/json',
      "Access-Control-Allow-Origin":'*' // Example: Setting the Content-Type header
    })
  };
  constructor(private http:HttpClient) { }
  uri = 'http://34.69.111.69:8000';
  //uri = 'http://localhost:8000'
  getPredictorInputs(){
    return this.http.get(`${this.uri}/api/predictor`,this.httpOptions);
  }
  submitPredictorInputs(predictorUserInputs:any){
    var body = predictorUserInputs
    return this.http.post(`${this.uri}/api/predictor`,body, this.httpOptions);
  }
  generateRespCurves() {
    return this.http.get(`${this.uri}/api/predictor/generate_response_curves`, this.httpOptions);
  }

  generateRespCurveMetrics() {
    return this.http.get(`${this.uri}/api/predictor/generate_response_metrics`, this.httpOptions);
  }
}
