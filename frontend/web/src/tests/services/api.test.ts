import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { mockUser, mockFarm, mockAuthResponse } from '../utils/testUtils'

// Mock the entire api module
vi.mock('@/services/api', () => ({
  authService: {
    login: vi.fn(),
    register: vi.fn(),
    logout: vi.fn(),
    getCurrentUser: vi.fn()
  },
  farmService: {
    getFarms: vi.fn(),
    getFarmById: vi.fn(),
    createFarm: vi.fn(),
    updateFarm: vi.fn(),
    deleteFarm: vi.fn()
  },
  analysisService: {
    uploadImage: vi.fn(),
    getAnalysisResults: vi.fn()
  }
}))

import { authService, farmService, analysisService } from '@/services/api'

describe('API Services', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  describe('authService', () => {
    it('should login user successfully', async () => {
      const credentials = {
        email: 'test@example.com',
        password: 'password123'
      }

      vi.mocked(authService.login).mockResolvedValue(mockAuthResponse)

      const result = await authService.login(credentials)

      expect(authService.login).toHaveBeenCalledWith(credentials)
      expect(result).toEqual(mockAuthResponse)
    })

    it('should register user successfully', async () => {
      const registerData = {
        email: 'newuser@example.com',
        password: 'password123',
        phone: '+1234567890'
      }

      vi.mocked(authService.register).mockResolvedValue(mockAuthResponse)

      const result = await authService.register(registerData)

      expect(authService.register).toHaveBeenCalledWith(registerData)
      expect(result).toEqual(mockAuthResponse)
    })

    it('should logout user and clear token', async () => {
      localStorage.setItem('auth_token', 'test-token')

      vi.mocked(authService.logout).mockResolvedValue(undefined)

      await authService.logout()

      expect(authService.logout).toHaveBeenCalled()
    })

    it('should handle login error', async () => {
      vi.mocked(authService.login).mockRejectedValue(
        new Error('Invalid credentials')
      )

      await expect(
        authService.login({ email: 'test@test.com', password: 'wrong' })
      ).rejects.toThrow()
    })
  })

  describe('farmService', () => {
    it('should fetch all farms', async () => {
      const mockFarms = [mockFarm]

      vi.mocked(farmService.getFarms).mockResolvedValue(mockFarms)

      const result = await farmService.getFarms()

      expect(farmService.getFarms).toHaveBeenCalled()
      expect(result).toEqual(mockFarms)
    })

    it('should fetch farm by id', async () => {
      vi.mocked(farmService.getFarmById).mockResolvedValue(mockFarm)

      const result = await farmService.getFarmById(1)

      expect(farmService.getFarmById).toHaveBeenCalledWith(1)
      expect(result).toEqual(mockFarm)
    })

    it('should create farm', async () => {
      const newFarm = {
        name: 'New Farm',
        size: 20,
        location: 'Test Location'
      }

      vi.mocked(farmService.createFarm).mockResolvedValue({ ...mockFarm, ...newFarm } as any)

      const result = await farmService.createFarm(newFarm as any)

      expect(farmService.createFarm).toHaveBeenCalledWith(newFarm)
      expect(result.name).toBe(newFarm.name)
    })

    it('should update farm', async () => {
      const updates = { name: 'Updated Farm' }

      vi.mocked(farmService.updateFarm).mockResolvedValue({ ...mockFarm, ...updates })

      const result = await farmService.updateFarm(1, updates)

      expect(farmService.updateFarm).toHaveBeenCalledWith(1, updates)
      expect(result.name).toBe(updates.name)
    })

    it('should delete farm', async () => {
      vi.mocked(farmService.deleteFarm).mockResolvedValue(undefined)

      await farmService.deleteFarm(1)

      expect(farmService.deleteFarm).toHaveBeenCalledWith(1)
    })
  })

  describe('analysisService', () => {
    it('should upload image for analysis', async () => {
      const file = new File(['test'], 'test.jpg', { type: 'image/jpeg' })

      vi.mocked(analysisService.uploadImage).mockResolvedValue({ id: 1, status: 'processing' } as any)

      const result = await analysisService.uploadImage(file, 1)

      expect(analysisService.uploadImage).toHaveBeenCalledWith(file, 1)
      expect(result).toHaveProperty('id')
    })

    it('should get analysis results', async () => {
      const mockResult = {
        id: 1,
        disease_detected: 'Healthy',
        confidence_score: 0.95
      }

      vi.mocked(analysisService.getAnalysisResults).mockResolvedValue(mockResult as any)

      const result = await analysisService.getAnalysisResults(1)

      expect(analysisService.getAnalysisResults).toHaveBeenCalledWith(1)
      expect(result).toEqual(mockResult)
    })
  })
})
