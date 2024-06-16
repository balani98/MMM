import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ExplorerFormComponent } from './explorer-form.component';

describe('ExplorerFormComponent', () => {
  let component: ExplorerFormComponent;
  let fixture: ComponentFixture<ExplorerFormComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [ExplorerFormComponent]
    });
    fixture = TestBed.createComponent(ExplorerFormComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
