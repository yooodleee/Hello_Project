/**@type {number} */
let amount;

amount = '12';
//      ^-- 'string' 형식의 인수는 'number' 형식의
//           매개변수에 할당될 수 없습니다.ts(2345)

/**
 * Adds VAT to price
 * 
 * @param {number} price The price without VAT
 * @param {number} vat The VAT [0-1]
 * 
 * @returns {number}
 */
function addVAT(price, vat = 0.2) {
    return price * (1 + vat);
}