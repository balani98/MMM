## Best Practices to Follow

### Version Control
- **Code and Documentation**:
  - Every code change must be documented and set up with version control systems.
  - For NAMER / UK projects, use the Brainlabs GitHub organization.

### Standard Git Practices
- **Branching**:
  - Fork your branch from the master branch of the repository.
- **Commits**:
  - Split your work into different commits for better organization and tracking.
- **Pull Requests**:
  - After pushing your code, raise a Pull Request (PR) and request a review from your reporting manager or a peer.
  - If your manager cannot be added as the reviewer, discuss the changes on a call. Make necessary changes as suggested, then add Hive / client as the reviewer for the PR. 
  - Upon approval of the PR, the changes can be merged.
- **Testing**:
  - Include possible test cases in your code to verify its correctness.

### Branch Management
- **Branch Types**:
  - Ideally, maintain three branches in GitHub: production, staging, and development.
  - Push and merge your code to the development branch for testing. If the testing is successful, create a pull request to the staging branch.
  - After verification on the staging server and approval from Hive / client, merge changes into the production branch.
  - If only two branches exist, manage the workflow accordingly.

### Communication and Coordination
- **Downtime Notification**:
  - If your application or data pipelines need to be shut down temporarily, inform your manager, Hive / lead, and the relevant Slack channel (e.g., `ask_platform_tech` or `ask_tech`).
  - Proceed only after receiving approval.
  
### Documentation and Project Management
- **Readme**:
  - Ensure the README file is properly documented upon project completion, including application and data pipeline architecture.
- **Monday.com**:
  - Ensure the project is tracked on Monday.com and that a ticket is created and updated regularly.
