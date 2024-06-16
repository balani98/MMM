import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PredictorMainComponent } from './predictor-main.component';

describe('PredictorMainComponent', () => {
  let component: PredictorMainComponent;
  let fixture: ComponentFixture<PredictorMainComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [PredictorMainComponent]
    });
    fixture = TestBed.createComponent(PredictorMainComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
