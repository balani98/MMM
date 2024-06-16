import { Component ,Input} from '@angular/core';
import { ExplorerService } from 'src/services/explorer.service';

@Component({
  selector: 'app-validation-report-stats',
  templateUrl: './validation-report-stats.component.html',
  styleUrls: ['./validation-report-stats.component.scss']
})
export class ValidationReportStatsComponent {
  constructor(private explorerService:ExplorerService){}
  @Input({required: true}) validationReport!: any;

}
