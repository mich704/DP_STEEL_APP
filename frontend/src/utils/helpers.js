export function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
      const cookies = document.cookie.split(';');
      for (let i = 0; i < cookies.length; i++) {
        const cookie = cookies[i].trim();
        // Does this cookie string begin with the name we want?
        if (cookie.substring(0, name.length + 1) === (name + '=')) {
          cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
          break;
        }
      }
    }
    return cookieValue;
}

export async function fetchGetRequestFromApi(url) {
  try {
    const response = await fetch(url);
    const data = await response.json();
    console.log(`Response from ${url}:`, data);
  } catch (error) {
    console.error(`Error fetching data from ${url}:`, error);
  }
}

export const acceptedImages = {
  'image/ras': ['.ras'],
  'image/x-xwindowdump': ['.xwd'],
  'image/bmp': ['.bmp'],
  'image/jpeg': ['.jpe', '.jpg', '.jpeg'],
  'image/x-xpixmap': ['.xpm'],
  'image/ief': ['.ief'],
  'image/x-portable-bitmap': ['.pbm'],
  'image/tiff': ['.tif', '.tiff'],
  'image/x-portable-pixmap': ['.ppm'],
  'image/x-xbitmap': ['.xbm'],
  'image/rgb': ['.rgb'],
  'image/x-portable-graymap': ['.pgm'],
  'image/png': ['.png'],
  'image/x-portable-anymap': ['.pnm'],
}



