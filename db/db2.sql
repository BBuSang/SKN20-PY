select * from usertbl where name = '박보검';

select * from usertbl where height > (select height from usertbl where name = "박보검");

-- 1. usertbl의 모든 데이터를 조회
select * from usertbl;
-- 2. 성이 김씨성을 가지는 사람을 모두 출력
select * from usertbl where name like "김%";
-- 3. 박씨 성을 가진 사람중에서 서울에 사는 사람
select * from usertbl where name like "박%" and addr = "서울";
-- 4. 이씨 성을 가진 사람들 중에서 지역이 서울이 아닌 사람들
select * from usertbl where name like "이%" and not addr like "서울";
-- 5. 1980년 이후 출생자 이면서 키가 170이상이고 성씨가 박씨인 사람들
select * from usertbl where name like "박%" and height >= 170 and birthyear > 1980;


-- any : 서브 쿼리의 결과물에 하나라도 만족
-- all : 서브 쿼리의 결과물에 모두 만족
select *
from usertbl
where height >
any(select height from usertbl where addr = '서울');

