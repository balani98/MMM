import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PredictorFormComponent } from './predictor-form.component';

describe('PredictorFormComponent', () => {
  let component: PredictorFormComponent;
  let fixture: ComponentFixture<PredictorFormComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [PredictorFormComponent]
    });
    fixture = TestBed.createComponent(PredictorFormComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
