import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ChannelProfilingStatsComponent } from './channel-profiling-stats.component';

describe('ChannelProfilingStatsComponent', () => {
  let component: ChannelProfilingStatsComponent;
  let fixture: ComponentFixture<ChannelProfilingStatsComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [ChannelProfilingStatsComponent]
    });
    fixture = TestBed.createComponent(ChannelProfilingStatsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
