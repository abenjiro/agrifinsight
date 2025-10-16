import Swal from 'sweetalert2'

// Success alert
export const showSuccess = (message: string, title: string = 'Success!') => {
  return Swal.fire({
    icon: 'success',
    title: title,
    text: message,
    confirmButtonColor: '#10b981', // green-500
    confirmButtonText: 'OK',
    timer: 3000,
    timerProgressBar: true,
  })
}

// Error alert
export const showError = (message: string, title: string = 'Error!') => {
  return Swal.fire({
    icon: 'error',
    title: title,
    text: message,
    confirmButtonColor: '#ef4444', // red-500
    confirmButtonText: 'OK',
  })
}

// Warning alert
export const showWarning = (message: string, title: string = 'Warning!') => {
  return Swal.fire({
    icon: 'warning',
    title: title,
    text: message,
    confirmButtonColor: '#f59e0b', // amber-500
    confirmButtonText: 'OK',
  })
}

// Info alert
export const showInfo = (message: string, title: string = 'Info') => {
  return Swal.fire({
    icon: 'info',
    title: title,
    text: message,
    confirmButtonColor: '#3b82f6', // blue-500
    confirmButtonText: 'OK',
  })
}

// Confirmation dialog
export const showConfirm = (
  message: string,
  title: string = 'Are you sure?',
  confirmText: string = 'Yes',
  cancelText: string = 'No'
) => {
  return Swal.fire({
    icon: 'question',
    title: title,
    text: message,
    showCancelButton: true,
    confirmButtonColor: '#10b981', // green-500
    cancelButtonColor: '#6b7280', // gray-500
    confirmButtonText: confirmText,
    cancelButtonText: cancelText,
  })
}

// Loading alert
export const showLoading = (message: string = 'Please wait...', title: string = 'Loading') => {
  return Swal.fire({
    title: title,
    text: message,
    allowOutsideClick: false,
    allowEscapeKey: false,
    didOpen: () => {
      Swal.showLoading()
    },
  })
}

// Close any open alert
export const closeAlert = () => {
  Swal.close()
}

// Toast notification (small popup)
export const showToast = (message: string, icon: 'success' | 'error' | 'warning' | 'info' = 'success') => {
  return Swal.fire({
    toast: true,
    position: 'top-end',
    icon: icon,
    title: message,
    showConfirmButton: false,
    timer: 3000,
    timerProgressBar: true,
  })
}
