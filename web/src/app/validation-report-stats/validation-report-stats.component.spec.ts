import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ValidationReportStatsComponent } from './validation-report-stats.component';

describe('ValidationReportStatsComponent', () => {
  let component: ValidationReportStatsComponent;
  let fixture: ComponentFixture<ValidationReportStatsComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [ValidationReportStatsComponent]
    });
    fixture = TestBed.createComponent(ValidationReportStatsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
