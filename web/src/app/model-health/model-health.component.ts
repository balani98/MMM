import { Component } from '@angular/core';
import { PredictorService } from 'src/services/predictor.service';

@Component({
  selector: 'app-model-health',
  templateUrl: './model-health.component.html',
  styleUrl: './model-health.component.scss'
})
export class ModelHealthComponent{
  predictorMetricsInputs = false
  responseMetrics:any={}
  constructor(private predictorService: PredictorService) {} 
  
  getResponseMetrics(){
    this.predictorService.generateRespCurveMetrics().subscribe({
      next: async(res: any) => {
        if(res.status == 200){
          this.predictorMetricsInputs = true
          this.responseMetrics = res.body.metrics
        }
      }, error: (errorResponse) => {

        if(errorResponse.status == 500)
          alert("No model exists for showing results")
        else if(errorResponse.status == 400)
          alert("Model is in training phase")
      }
    });
  }
  ngOnInit(): void {
    console.log('ngOnInit called'); // Check if this is logged
    this.getResponseMetrics()
  }

}
