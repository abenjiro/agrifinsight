import { ReactElement } from 'react'
import { render, RenderOptions } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import { SidebarProvider } from '@/contexts/SidebarContext'

interface AllTheProvidersProps {
  children: React.ReactNode
}

function AllTheProviders({ children }: AllTheProvidersProps) {
  return (
    <BrowserRouter>
      <SidebarProvider>
        {children}
      </SidebarProvider>
    </BrowserRouter>
  )
}

const customRender = (
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
) => render(ui, { wrapper: AllTheProviders, ...options })

export * from '@testing-library/react'
export { customRender as render }

// Mock data utilities
export const mockUser = {
  id: 1,
  email: 'test@example.com',
  phone: '+1234567890',
  role: 'farmer',
  is_active: true,
  created_at: new Date().toISOString()
}

export const mockFarm = {
  id: 1,
  user_id: 1,
  name: 'Test Farm',
  address: '123 Farm Road',
  latitude: 5.6037,
  longitude: -0.1870,
  size: 10.5,
  size_unit: 'acres',
  soil_type: 'Loamy',
  created_at: new Date().toISOString()
}

export const mockCropImage = {
  id: 1,
  farm_id: 1,
  field_id: 1,
  image_url: '/uploads/test.jpg',
  filename: 'test.jpg',
  analysis_status: 'completed',
  uploaded_at: new Date().toISOString()
}

export const mockAnalysisResult = {
  id: 1,
  image_id: 1,
  disease_detected: 'Healthy',
  confidence_score: 0.95,
  disease_type: 'None',
  severity: 'none',
  recommendations: 'Continue current care',
  health_score: 95.0,
  created_at: new Date().toISOString()
}

export const mockAuthResponse = {
  user: mockUser,
  access_token: 'mock-jwt-token',
  refresh_token: 'mock-refresh-token',
  token_type: 'bearer'
}
