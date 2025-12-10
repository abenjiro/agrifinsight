# Frontend Testing Documentation

## Quick Start

```bash
# Install dependencies
npm install

# Run all tests
npm test

# Run with UI
npm run test:ui

# Run with coverage
npm run test:coverage
```

## Test Organization

### Test Structure
```
src/tests/
├── setup.ts              # Global test setup
├── utils/
│   └── testUtils.tsx    # Custom render & mocks
├── components/          # Component tests
├── pages/               # Page tests
├── services/            # API service tests
└── integration/         # Integration tests
```

### Test Types
1. **Component Tests** - Individual UI components
2. **Page Tests** - Full page functionality
3. **Service Tests** - API interactions
4. **Integration Tests** - End-to-end flows

## Writing Tests

### Component Test Example

```typescript
import { render, screen, fireEvent } from '../utils/testUtils'
import MyButton from '@/components/MyButton'

describe('MyButton', () => {
  it('should handle click', () => {
    const onClick = vi.fn()
    render(<MyButton onClick={onClick}>Click Me</MyButton>)

    fireEvent.click(screen.getByText('Click Me'))

    expect(onClick).toHaveBeenCalled()
  })
})
```

### API Test Example

```typescript
import { vi } from 'vitest'
import { authService } from '@/services/api'
import axios from 'axios'

vi.mock('axios')

describe('authService', () => {
  it('should login user', async () => {
    vi.mocked(axios.post).mockResolvedValue({
      data: { token: 'abc123' }
    })

    const result = await authService.login({
      email: 'test@test.com',
      password: 'pass'
    })

    expect(result.token).toBe('abc123')
  })
})
```

## Testing Utilities

### Custom Render
Automatically wraps components with providers:
```typescript
import { render } from '@/tests/utils/testUtils'

render(<MyComponent />)
// Includes Router, Context providers, etc.
```

### Mock Data
Pre-defined mock objects:
- `mockUser`
- `mockFarm`
- `mockCropImage`
- `mockAnalysisResult`
- `mockAuthResponse`

### Query Methods (Priority Order)
1. `getByRole` - Most accessible
2. `getByLabelText` - Form elements
3. `getByText` - Text content
4. `getByTestId` - Last resort

## Best Practices

### Do ✅
- Test user behavior, not implementation
- Use semantic queries (`getByRole`)
- Test error states and loading
- Mock external APIs
- Keep tests isolated

### Don't ❌
- Test internal state
- Use implementation details
- Share state between tests
- Test third-party libraries
- Make tests dependent on each other

## Common Patterns

### Testing Async Operations
```typescript
await waitFor(() => {
  expect(screen.getByText('Loaded')).toBeInTheDocument()
})
```

### Testing Forms
```typescript
const emailInput = screen.getByLabelText(/email/i)
fireEvent.change(emailInput, { target: { value: 'test@test.com' } })
```

### Testing Navigation
```typescript
const link = screen.getByRole('link', { name: /dashboard/i })
fireEvent.click(link)

await waitFor(() => {
  expect(window.location.pathname).toBe('/dashboard')
})
```

## Coverage Goals

- Components: >70%
- Pages: >75%
- Services: >80%
- Overall: >70%

## Debugging Tests

### Run Specific Test
```bash
npm test -- Header.test.tsx
```

### Watch Mode
```bash
npm test -- --watch
```

### Debug in Browser
```bash
npm run test:ui
```

### View Coverage
```bash
npm run test:coverage
open coverage/index.html
```

## CI/CD

Tests run automatically:
- On every commit
- In pull requests
- Before deployment

Ensure all tests pass before merging.
