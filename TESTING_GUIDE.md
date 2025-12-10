# AgriFinSight Testing Guide

This guide provides comprehensive instructions for running and writing tests for the AgriFinSight platform.

## Table of Contents
- [Backend Testing](#backend-testing)
- [Frontend Testing](#frontend-testing)
- [Test Coverage](#test-coverage)
- [Writing Tests](#writing-tests)
- [CI/CD Integration](#cicd-integration)

---

## Backend Testing

### Overview
The backend uses **pytest** for testing with the following structure:
- Unit tests for database models
- API endpoint tests
- Service layer tests
- Integration tests

### Test Structure
```
backend/
├── conftest.py              # Pytest configuration and fixtures
├── tests/
│   ├── __init__.py
│   ├── test_models.py       # Database model tests
│   ├── test_auth.py         # Authentication endpoint tests
│   ├── test_farms.py        # Farm management tests
│   ├── test_analysis.py     # Crop analysis tests
│   └── test_services.py     # Service layer tests
```

### Running Backend Tests

#### Prerequisites
1. Ensure you're in the backend directory:
   ```bash
   cd backend
   ```

2. Activate virtual environment:
   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install test dependencies (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

#### Run All Tests
```bash
pytest
```

#### Run with Verbose Output
```bash
pytest -v
```

#### Run Specific Test File
```bash
pytest tests/test_auth.py
```

#### Run Specific Test Class or Function
```bash
pytest tests/test_auth.py::TestAuthEndpoints
pytest tests/test_auth.py::TestAuthEndpoints::test_login_success
```

#### Run with Coverage Report
```bash
pytest --cov=app --cov-report=html --cov-report=term
```

This generates:
- Terminal coverage report
- HTML coverage report in `htmlcov/` directory

#### Run Tests in Parallel (faster)
```bash
pip install pytest-xdist
pytest -n auto
```

### Backend Test Fixtures

Common fixtures available in `conftest.py`:

- **`db`**: Fresh database session for each test
- **`client`**: FastAPI test client with database override
- **`test_user`**: Pre-created test user
- **`test_user_token`**: JWT token for test user
- **`auth_headers`**: Authorization headers with token
- **`test_farm`**: Pre-created test farm
- **`test_field`**: Pre-created test field
- **`test_crop_image`**: Pre-created crop image

#### Example Usage
```python
def test_get_farm(client, auth_headers, test_farm):
    response = client.get(
        f"/api/farms/{test_farm.id}",
        headers=auth_headers
    )
    assert response.status_code == 200
```

---

## Frontend Testing

### Overview
The frontend uses:
- **Vitest** as the test runner
- **React Testing Library** for component testing
- **jsdom** for DOM simulation

### Test Structure
```
frontend/web/src/tests/
├── setup.ts                    # Test setup and global mocks
├── utils/
│   └── testUtils.tsx          # Custom render and mock data
├── components/
│   ├── Header.test.tsx
│   ├── DashboardSidebar.test.tsx
│   └── ProtectedRoute.test.tsx
├── pages/
│   └── LoginPage.test.tsx
├── services/
│   └── api.test.ts
└── integration/
    ├── auth.test.tsx
    └── farmManagement.test.tsx
```

### Running Frontend Tests

#### Prerequisites
1. Navigate to frontend directory:
   ```bash
   cd frontend/web
   ```

2. Install dependencies (if not already done):
   ```bash
   npm install
   ```

#### Run All Tests
```bash
npm test
```

#### Run in Watch Mode
```bash
npm test -- --watch
```

#### Run with UI (interactive)
```bash
npm run test:ui
```

#### Run with Coverage
```bash
npm run test:coverage
```

Coverage report will be in `coverage/` directory.

#### Run Specific Test File
```bash
npm test -- src/tests/components/Header.test.tsx
```

#### Run Tests Matching Pattern
```bash
npm test -- --grep="Login"
```

### Frontend Testing Utilities

Custom render function from `testUtils.tsx`:
```typescript
import { render, screen } from '@/tests/utils/testUtils'

// Automatically wraps components with providers
render(<MyComponent />)
```

Mock data available:
- `mockUser`
- `mockFarm`
- `mockCropImage`
- `mockAnalysisResult`
- `mockAuthResponse`

---

## Test Coverage

### Backend Coverage Goals
- **Models**: >90% coverage
- **API Endpoints**: >85% coverage
- **Services**: >80% coverage
- **Overall**: >80% coverage

### Frontend Coverage Goals
- **Components**: >70% coverage
- **Pages**: >75% coverage
- **Services**: >80% coverage
- **Overall**: >70% coverage

### Viewing Coverage Reports

#### Backend
```bash
cd backend
pytest --cov=app --cov-report=html
open htmlcov/index.html  # macOS
```

#### Frontend
```bash
cd frontend/web
npm run test:coverage
open coverage/index.html  # macOS
```

---

## Writing Tests

### Backend Test Example

```python
# tests/test_example.py
import pytest
from app.models.database import Farm

class TestFarmEndpoints:
    """Test farm CRUD operations"""

    def test_create_farm(self, client, auth_headers):
        """Test creating a new farm"""
        response = client.post(
            "/api/farms",
            headers=auth_headers,
            json={
                "name": "My Farm",
                "size": 10.5,
                "size_unit": "acres"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "My Farm"
        assert data["size"] == 10.5

    def test_get_farm_unauthorized(self, client, test_farm):
        """Test accessing farm without authentication"""
        response = client.get(f"/api/farms/{test_farm.id}")
        assert response.status_code == 401
```

### Frontend Test Example

```typescript
// tests/components/MyComponent.test.tsx
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '../utils/testUtils'
import MyComponent from '@/components/MyComponent'

describe('MyComponent', () => {
  it('should render with props', () => {
    render(<MyComponent title="Test Title" />)

    expect(screen.getByText('Test Title')).toBeInTheDocument()
  })

  it('should handle click event', () => {
    const handleClick = vi.fn()
    render(<MyComponent onClick={handleClick} />)

    const button = screen.getByRole('button')
    fireEvent.click(button)

    expect(handleClick).toHaveBeenCalledOnce()
  })
})
```

### Best Practices

#### General
1. **Test behavior, not implementation**
   - Focus on what the user sees/does
   - Avoid testing internal state

2. **Keep tests isolated**
   - Each test should be independent
   - Use fixtures/mocks to avoid external dependencies

3. **Use descriptive test names**
   - `test_user_cannot_access_other_users_farm()`
   - Better than `test_farm_access()`

4. **Follow AAA pattern**
   - Arrange: Set up test data
   - Act: Execute the code
   - Assert: Verify results

#### Backend Specific
- Use database fixtures for consistent state
- Test both success and error cases
- Mock external services (weather API, AI models)
- Test authentication and authorization

#### Frontend Specific
- Use `screen.getByRole()` over `getByTestId()`
- Test user interactions, not implementation details
- Mock API calls with vi.mock()
- Test loading states and error handling

---

## CI/CD Integration

### GitHub Actions Example

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  backend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          cd backend
          pip install -r requirements.txt

      - name: Run tests
        run: |
          cd backend
          pytest --cov=app --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./backend/coverage.xml

  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Node
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install dependencies
        run: |
          cd frontend/web
          npm install

      - name: Run tests
        run: |
          cd frontend/web
          npm run test:coverage

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./frontend/web/coverage/coverage-final.json
```

### Pre-commit Hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash

echo "Running tests before commit..."

# Backend tests
cd backend
pytest
if [ $? -ne 0 ]; then
    echo "Backend tests failed. Commit aborted."
    exit 1
fi

# Frontend tests
cd ../frontend/web
npm test -- --run
if [ $? -ne 0 ]; then
    echo "Frontend tests failed. Commit aborted."
    exit 1
fi

echo "All tests passed!"
exit 0
```

Make it executable:
```bash
chmod +x .git/hooks/pre-commit
```

---

## Troubleshooting

### Backend Issues

**Issue**: Database errors in tests
```bash
# Solution: Ensure test database is isolated
pytest --create-db
```

**Issue**: Import errors
```bash
# Solution: Ensure PYTHONPATH is set
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest
```

### Frontend Issues

**Issue**: Module not found errors
```bash
# Solution: Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

**Issue**: Tests timing out
```javascript
// Solution: Increase timeout in vitest.config.ts
export default defineConfig({
  test: {
    testTimeout: 10000
  }
})
```

---

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Vitest Documentation](https://vitest.dev/)
- [React Testing Library](https://testing-library.com/react)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)

---

## Contributing

When adding new features:
1. Write tests first (TDD approach recommended)
2. Ensure tests pass locally
3. Maintain or improve code coverage
4. Update this guide if adding new test patterns

**Last Updated**: December 2024
