import { AfterViewInit, Component, ElementRef, Input, OnInit } from '@angular/core';
import Chart from 'chart.js/auto';
import {
  faDownload
} from '@fortawesome/free-solid-svg-icons';
import { ExplorerService } from 'src/services/explorer.service';
@Component({
  selector: 'app-sales-by-month',
  templateUrl: './sales-by-month.component.html',
  styleUrls: ['./sales-by-month.component.scss'],
})
export class SalesByMonthComponent implements OnInit {
  constructor(private explorerService:ExplorerService){}
  public chart: any;
  faDownload = faDownload;
  @Input({required: true}) histogramStats!: any;
  @Input({required: true}) variableName!: any;
  createChart(){
    console.log(this.histogramStats);
    const x = this.histogramStats?.bin_edges;
    const y = this.histogramStats?.counts;
    // Calculate class interval
      const classInterval = x[1] - x[0];

    // Calculate bin edges for class intervals
    const binEdges = x.map((edge:any) => edge.toFixed(2));
  
    this.chart = new Chart('MyChart', {
      type: 'bar',
      data: {
        labels:binEdges, // Convert x values to strings for labels
        datasets: [{
          label: '',
          data: y,
          backgroundColor: '#FA6C97',
          borderWidth: 1,
          barPercentage:1,
          categoryPercentage:1
        }]
      },
      options: {
        plugins: {
          legend: {
              display: false
          },
       },
        scales: {
          x: {
            title: {
              display: true,
              text: this.variableName
            },
            ticks: {
              autoSkip: true,
              maxTicksLimit: 20, // Adjust the number of ticks as needed
            },
          },
          y: {
            title: {
              display: true,
              text: 'Frequency'
            }
          }
        }
      }
    });
  }
  downloadDetailedEDAReport(){
    this.explorerService.downloadEDAreport().subscribe(
      (htmlContent: string) => {
        this.downloadFile(htmlContent, 'EDA.html', 'text/html');
      })
  }
  private downloadFile(data: string, filename: string, mimeType: string) {
    const blob = new Blob([data], { type: mimeType });

    const anchor = document.createElement('a');
    anchor.href = window.URL.createObjectURL(blob);
    anchor.download = filename;
    anchor.style.display = 'none';

    document.body.appendChild(anchor);
    anchor.click();

    document.body.removeChild(anchor);
  }
  ngOnInit(): void {
    this.createChart()
  }
}
