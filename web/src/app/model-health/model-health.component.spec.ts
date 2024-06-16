import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ModelHealthComponent } from './model-health.component';

describe('ModelHealthComponent', () => {
  let component: ModelHealthComponent;
  let fixture: ComponentFixture<ModelHealthComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ModelHealthComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(ModelHealthComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
