import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ChannelQuantileStatsComponent } from './channel-quantile-stats.component';

describe('ChannelQuantileStatsComponent', () => {
  let component: ChannelQuantileStatsComponent;
  let fixture: ComponentFixture<ChannelQuantileStatsComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [ChannelQuantileStatsComponent]
    });
    fixture = TestBed.createComponent(ChannelQuantileStatsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
