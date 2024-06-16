import { Component, Input } from '@angular/core';
import { ShortNumberPipe } from 'src/pipes/shortnumber.pipe';
@Component({
  selector: 'app-profiling-stats',
  templateUrl: './profiling-stats.component.html',
  styleUrls: ['./profiling-stats.component.scss']
})
export class ProfilingStatsComponent {
  @Input({required: true}) overviewStats!: any;
}
