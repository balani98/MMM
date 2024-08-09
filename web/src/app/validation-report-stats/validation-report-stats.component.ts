import { Component ,Input} from '@angular/core';
import { ExplorerService } from 'src/services/explorer.service';
import { ScrollPanelModule } from 'primeng/scrollpanel';
@Component({
  selector: 'app-validation-report-stats',
  templateUrl: './validation-report-stats.component.html',
  styleUrls: ['./validation-report-stats.component.scss']
})
export class ValidationReportStatsComponent {
  constructor(private explorerService:ExplorerService){}
  @Input({required: true}) validationReport!: any;
  @Input({required: true}) validationReportZeroDataPoints!: any;
  @Input({required: true}) validationReportMisingDataPoints!: any;
  @Input({required: true}) validationReportOutliers!: any;
  @Input({required: true}) validationReportVarianceVariables!: any;

}
