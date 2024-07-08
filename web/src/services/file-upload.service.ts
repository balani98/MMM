import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders} from '@angular/common/http';
@Injectable({
  providedIn: 'root'
})
export class FileUploadService {
  //uri = 'https://global-mso-api-lr4grty6xq-uc.a.run.app';
  uri = 'http://34.69.111.69:8000';
  //uri = 'http://localhost:8000'
  constructor(private http:HttpClient) { }
  httpOptions = {
    headers: new HttpHeaders({
      "Access-Control-Allow-Origin":'*' // Example: Setting the Content-Type header
    })
  };
  uploadFile(formdata:FormData){
    console.log(formdata)
    return this.http.post(`${this.uri}/api/explorer/uploadfile`,formdata,this.httpOptions)
  }
}
