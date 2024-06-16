import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ProfilingStatsComponent } from './profiling-stats.component';

describe('ProfilingStatsComponent', () => {
  let component: ProfilingStatsComponent;
  let fixture: ComponentFixture<ProfilingStatsComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [ProfilingStatsComponent]
    });
    fixture = TestBed.createComponent(ProfilingStatsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
