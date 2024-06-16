import { Component, Input } from '@angular/core';
import { ExplorerService } from 'src/services/explorer.service';
import {
  faDownload
} from '@fortawesome/free-solid-svg-icons';
@Component({
  selector: 'app-channel-profiling-stats',
  templateUrl: './channel-profiling-stats.component.html',
  styleUrls: ['./channel-profiling-stats.component.scss']
})
export class ChannelProfilingStatsComponent {
  constructor(private explorerService:ExplorerService){}
  @Input({required: true}) variableStats!: any;
  faDownload = faDownload;

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
}



